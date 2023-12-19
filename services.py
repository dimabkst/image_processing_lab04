from math import exp, pi
from numpy import array, sqrt, complex128
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.signal import convolve2d
from typing import Literal, Union
from custom_types import DiscreteFourierTransform, DiscreteFunctionMatrix, ListImage, FloatOrNone, ImageFunction, FilterKernel, PSFKernerl, Sizes2D
from utils import convertToProperImage

def getMean(image: ListImage) -> float:
    N = len(image)
    M = len(image[0])

    mean = sum([sum([image[i][j] for j in range(M)]) for i in range(N)]) / (N * M)

    return mean

def getVariance(image: ListImage, mean: FloatOrNone=None) -> float:
    N = len(image)
    M = len(image[0])

    mean = mean if mean is not None else getMean(image)

    variance = sum([sum([(image[i][j] - mean) ** 2 for j in range(M)]) for i in range(N)]) / (N * M)

    return variance

def getStandardDeviation(image: ListImage, mean: FloatOrNone=None, variance: FloatOrNone=None) -> float:
    variance = variance if variance is not None else getVariance(image, mean)

    standard_deviation = variance ** 0.5

    return standard_deviation

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

def linearSpatialFiltering(image: ListImage, filter_kernel: FilterKernel) -> ListImage:
    N = len(image)
    M = len(image[0])

    filter_kernel_sizes = (len(filter_kernel), len(filter_kernel[0]))

    extended_image_function = getMirroredImageFunction(image, filter_kernel_sizes)

    a = filter_kernel_sizes[0] // 2 # equal to (filterKernelSizes[0] - 1) / 2 in formula
    b = filter_kernel_sizes[1] // 2

    filtered_image = [[sum([sum([filter_kernel[a + s][b + t] * extended_image_function(i + s, j + t) for t in range(-b, b + 1)]) for s in range(-a, a + 1)]) for j in range(M)] for i in range(N)]

    return convertToProperImage(filtered_image)

def get2DDiscreteFourierTransform(discrete_function: DiscreteFunctionMatrix, sizes: Union[Sizes2D, None] = None, centered: bool = True, normalized: bool = False) -> DiscreteFourierTransform:
    # Convert the function to a NumPy array for faster computation
    np_discrete_function = array(discrete_function, dtype=complex128)

    sizes = sizes or (len(discrete_function), len(discrete_function[0]))

    # Compute the 2D FFT using NumPy
    np_fft_result = fft2(np_discrete_function, s=sizes, norm='ortho')

    # center if needed
    np_result = fftshift(np_fft_result) if centered else np_fft_result

    # # normalize if needed
    # if (normalized):
    #     np_result = np_result / sqrt(len(discrete_function) * len(discrete_function[0]))

    result = np_result.tolist()

    return result

def get2DInverseDiscreteFourierTransform(fourier_transform: DiscreteFourierTransform, sizes: Union[Sizes2D, None] = None, centered: bool = True, normalized: bool = False) -> ListImage:
    # Convert the fourier transform to a NumPy array for faster computation
    np_fourier_transform = array(fourier_transform, dtype=complex128)

    # If dft is centered
    if centered:
        np_fourier_transform = ifftshift(np_fourier_transform)

    sizes = sizes or (len(fourier_transform), len(fourier_transform[0]))

    # Perform the inverse 2D FFT using NumPy
    np_inverse_fft_result = ifft2(np_fourier_transform, s=sizes, norm='ortho')

    # # denormalize if needed
    # if (normalized):
    #     np_inverse_fft_result = np_inverse_fft_result * sqrt(sizes[0] * sizes[1])
    
    # Convert the result to a ListImage
    inverse_fft_result = np_inverse_fft_result.real.tolist()

    return convertToProperImage(inverse_fft_result)

# own implementation of MATLAB fspecial('gaussian', window_size, sigma)
def getGaussianPSF(window_sizes: Sizes2D, sigma: float = 1.0, centered: bool = True, normalization: Literal['sum', 'pi'] = 'sum') -> PSFKernerl:
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

def blurrImage(image: ListImage, psf: PSFKernerl) -> ListImage:
    psf_sum = sum([sum(row) for row in psf])

    # normalize PSF
    normalized_psf = [[el / psf_sum for el in row] for row in psf]

    # MATLAB imfilter analogue
    blurred_image = convolve2d(image, normalized_psf, mode='same', boundary='symm').tolist()

    return convertToProperImage(blurred_image)

def wienerFiltration(blurred_image: ListImage, psf: PSFKernerl, alpha: float) -> ListImage:
    M = len(blurred_image)
    N = len(blurred_image[0])

    filter_kernel = createFilterKernel([[1, 2, 1], [2, 4, 2], [1, 2, 1]])

    filtered_image = linearSpatialFiltering(blurred_image, filter_kernel)

    noise_approximation = [[abs(blurred_image[i][j] - filtered_image[i][j]) for j in range(N)] for i in range(M)]

    K = getVariance(noise_approximation) / getVariance(blurred_image)

    blurred_image_fourier_transform = get2DDiscreteFourierTransform(blurred_image)

    psf_fourier_transform = get2DDiscreteFourierTransform(psf, sizes=(M, N))

    restored_image_fourier_transform = [[(psf_fourier_transform[u][v].conjugate() / (abs(psf_fourier_transform[u][v]) ** 2 + alpha * K)) * blurred_image_fourier_transform[u][v] 
                                         for v in range(N)] for u in range(M)]
    
    return get2DInverseDiscreteFourierTransform(restored_image_fourier_transform)