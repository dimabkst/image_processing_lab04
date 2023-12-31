from math import exp, log10, pi
from numpy import array, sqrt, complex128
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.signal import convolve2d
from typing import List, Literal, Union
from custom_types import DiscreteFourierTransform, DiscreteFunctionMatrix, ListImage, FloatOrNone, ImageFunction, FilterKernel, ListImageRaw, PSFKernerl, Sizes2D
from utils import convertToProperImage, generateGaussianNoise, plot2DMatrix

def getMean(image: Union[ListImage, ListImageRaw]) -> float:
    N = len(image)
    M = len(image[0])

    mean = sum([sum([image[i][j] for j in range(M)]) for i in range(N)]) / (N * M)

    return mean

def getVariance(image: Union[ListImage, ListImageRaw], mean: FloatOrNone=None) -> float:
    N = len(image)
    M = len(image[0])

    mean = mean if mean is not None else getMean(image)

    variance = sum([sum([(image[i][j] - mean) ** 2 for j in range(M)]) for i in range(N)]) / (N * M)

    return variance

def getStandardDeviation(image: Union[ListImage, ListImageRaw], mean: FloatOrNone=None, variance: FloatOrNone=None) -> float:
    variance = variance if variance is not None else getVariance(image, mean)

    standard_deviation = variance ** 0.5

    return standard_deviation

def getBSNR(blurred_image_without_noise: ListImage, restored_image: ListImage, noise_variance: float) -> float:
    M = len(blurred_image_without_noise)
    N = len(blurred_image_without_noise[0])

    return 10 * log10(sum([sum([(blurred_image_without_noise[i][j] - restored_image[i][j]) ** 2 for j in range(N)]) for i in range(M)]) / (M * N * noise_variance))

def getISNR(ideal_image: ListImage, degraded_image: ListImage, restored_image: ListImage) -> float:
    M = len(ideal_image)
    N = len(ideal_image[0])

    return 10 * log10(sum([sum([(ideal_image[i][j] - degraded_image[i][j]) ** 2 for j in range(N)]) for i in range(M)]) / sum([sum([(ideal_image[i][j] - restored_image[i][j]) ** 2 for j in range(N)]) for i in range(M)]))

def createFilterKernel(weights: FilterKernel) -> FilterKernel:
    filter_kernel = weights

    filter_kernel_sum = sum([sum(filter_kernel[row]) for row in range(len(filter_kernel))])

    if filter_kernel_sum != 1:
        filter_kernel = [[el / filter_kernel_sum for el in row] for row in filter_kernel]

    return filter_kernel

def getMirroredImageFunction(image: ListImage, filter_kernel_sizes: Sizes2D) -> ImageFunction:
    N = len(image)
    M = len(image[0])
    
    extension_sizes = tuple(size // 2 for size in filter_kernel_sizes)

    def mirroredImageFunction(i: int, j: int) -> int:
        if i >= N + extension_sizes[0]:
            raise KeyError
        elif i < 0:
            ii = -i
        elif i >= N:
            ii = N - 1 - (i - N + 1)
        else:
            ii = i

        if j >= M + extension_sizes[1]:
            raise KeyError
        elif j < 0:
            jj = -j
        elif j >= M:
            jj = M - 1 - (j - M + 1)
        else:
            jj = j

        return image[ii][jj]

    return mirroredImageFunction

def linearSpatialFilteringRaw(image: ListImage, filter_kernel: FilterKernel) -> ListImageRaw:
    N = len(image)
    M = len(image[0])

    filter_kernel_sizes = (len(filter_kernel), len(filter_kernel[0]))

    extended_image_function = getMirroredImageFunction(image, filter_kernel_sizes)

    a = filter_kernel_sizes[0] // 2 # equal to (filterKernelSizes[0] - 1) / 2 in formula
    b = filter_kernel_sizes[1] // 2

    filtered_image = [[sum([sum([filter_kernel[a + s][b + t] * extended_image_function(i + s, j + t) for t in range(-b, b + 1)]) for s in range(-a, a + 1)]) for j in range(M)] for i in range(N)]

    return filtered_image

def linearSpatialFiltering(image: ListImage, filter_kernel: FilterKernel) -> ListImage:
    filtered_image = linearSpatialFilteringRaw(image, filter_kernel)

    return convertToProperImage(filtered_image)

def get2DDiscreteFourierTransform(discrete_function: DiscreteFunctionMatrix, sizes: Union[Sizes2D, None] = None, centered: bool = False, normalized: bool = False) -> DiscreteFourierTransform:
    # Convert the function to a NumPy array for faster computation
    np_discrete_function = array(discrete_function, dtype=complex128)

    sizes = sizes or (len(discrete_function), len(discrete_function[0]))

    # Compute the 2D FFT using NumPy
    np_fft_result = fft2(np_discrete_function, s=sizes)

    # center if needed
    np_result = fftshift(np_fft_result) if centered else np_fft_result

    # normalize if needed
    if (normalized):
        np_result = np_result / sqrt(len(discrete_function) * len(discrete_function[0]))

    result = np_result.tolist()

    return result

def get2DInverseDiscreteFourierTransform(fourier_transform: DiscreteFourierTransform, centered: bool = False, normalized: bool = False) -> ListImage:
    # Convert the fourier transform to a NumPy array for faster computation
    np_fourier_transform = array(fourier_transform, dtype=complex128)

    # If dft is centered
    if centered:
        np_fourier_transform = ifftshift(np_fourier_transform)

    # Perform the inverse 2D FFT using NumPy
    np_inverse_fft_result = ifft2(np_fourier_transform)

    # denormalize if needed
    if (normalized):
        np_inverse_fft_result = np_inverse_fft_result * sqrt(len(fourier_transform) * len(fourier_transform[0]))
    
    # Convert the result to a ListImage
    inverse_fft_result = abs(np_inverse_fft_result).tolist()

    return convertToProperImage(inverse_fft_result)

def get2DDiscreteFourierTransformMagnitude(fourier_transform: DiscreteFourierTransform) -> List[List[float]]:
    return [[abs(el) for el in row] for row in fourier_transform]

def plot2DDiscreteFourierTransform(fourier_transform: DiscreteFourierTransform, centered: bool = True) -> None:
    if centered:
        fourier_transform = fftshift(array(fourier_transform)).tolist()
        
    plot2DMatrix(get2DDiscreteFourierTransformMagnitude(fourier_transform))

def addGaussianAdditiveNoise(image: ListImage, std_dev_coef: float) -> ListImage:
    N = len(image)
    M = len(image[0])

    mean = getMean(image)

    standart_deviation = getStandardDeviation(image, mean)

    noise = generateGaussianNoise(mean, std_dev_coef * standart_deviation, (N, M))

    noisy_image = [[image[i][j] + noise[i][j] - mean for j in range(M)] for i in range(N)]

    return convertToProperImage(noisy_image)

# own implementation of MATLAB fspecial('gaussian', window_size, sigma)
def getGaussianPSF(window_sizes: Sizes2D, sigma: float = 1.0, centered: bool = False, normalization: Literal['sum', 'pi'] = 'sum') -> PSFKernerl:
    center_x = 0
    center_y = 0

    # for centering PSF properly
    if (centered):
        center_x = (window_sizes[0] - 1) // 2
        center_y = (window_sizes[1] - 1) // 2

    sigma_square = sigma ** 2

    psf = [[exp(-((x - center_x) ** 2 + (y - center_y) ** 2) / (2 * sigma_square)) for y in range(window_sizes[1])] for x in range(window_sizes[0])]

    denominator = 2 * pi * sigma_square if normalization == 'pi' else sum([sum(row) for row in psf])

    normalized_psf =  [[el / denominator for el in row] for row in psf]

    return normalized_psf

def _blurrImage(image: ListImage, psf: PSFKernerl) -> ListImage:
    psf_sum = sum([sum(row) for row in psf])

    # normalize PSF
    normalized_psf = [[el / psf_sum for el in row] for row in psf]

    # MATLAB imfilter analogue
    blurred_image = convolve2d(image, normalized_psf, mode='same', boundary='symm').tolist()

    return convertToProperImage(blurred_image)

def blurrImage(image: ListImage, psf: PSFKernerl) -> ListImage:
    M = len(image)
    N = len(image[0])

    image_fourier_transform = get2DDiscreteFourierTransform(image)

    psf_fourier_transform = get2DDiscreteFourierTransform(psf)

    blurred_image_fourier_transform = [[image_fourier_transform[i][j] * psf_fourier_transform[i][j] for j in range(N)] for i in range(M)]

    return get2DInverseDiscreteFourierTransform(blurred_image_fourier_transform)

def _getNoiseVariance(blurred_image: ListImage) -> float:
    M = len(blurred_image)
    N = len(blurred_image[0])

    filter_kernel = createFilterKernel([[1, 2, 1], [2, 4, 2], [1, 2, 1]])

    filtered_image = linearSpatialFilteringRaw(blurred_image, filter_kernel)

    noise_approximation = [[abs(blurred_image[i][j] - filtered_image[i][j]) for j in range(N)] for i in range(M)]

    noise_variance = getVariance(noise_approximation)
    
    return noise_variance

def wienerFiltration(blurred_image: ListImage, psf: PSFKernerl, alpha: float) -> ListImage:
    M = len(blurred_image)
    N = len(blurred_image[0])
    
    noise_variance = _getNoiseVariance(blurred_image)

    blurred_variance = getVariance(blurred_image)

    K = M * N * noise_variance / blurred_variance

    blurred_image_fourier_transform = get2DDiscreteFourierTransform(blurred_image)

    psf_fourier_transform = get2DDiscreteFourierTransform(psf)
    
    restored_image_fourier_transform = [[(psf_fourier_transform[u][v].conjugate() / (abs(psf_fourier_transform[u][v]) ** 2 + alpha * K)) * blurred_image_fourier_transform[u][v] 
                                         for v in range(N)] for u in range(M)]
    
    return get2DInverseDiscreteFourierTransform(restored_image_fourier_transform)