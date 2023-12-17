from cmath import exp, pi
from typing import Tuple
from custom_types import DiscreteFourierTransform, DiscreteFourierTransformFunction, ListImage
from numpy.fft import fft2, ifft2
from numpy import array, complex128
from utils import convertToProperImage

# deprecated, too slow
def get2DDiscreteFourierTransformFunction(image: ListImage) -> DiscreteFourierTransformFunction:
    M = len(image)
    N = len(image[0])

    def discreteFourierTransform(u: int, v: int):
        return sum([sum([image[x][y] * exp(-2j * pi * (u * x / M + v * y / N)) for y in range(N)]) for x in range(M)]) / (M * N)
    
    return discreteFourierTransform

# not normalized
def get2DDiscreteFourierTransform(image: ListImage) -> DiscreteFourierTransform:
    M = len(image)
    N = len(image[0])

    # Convert the image to a NumPy array for faster computation
    np_image = array(image, dtype=complex128)

    # Compute the 2D FFT using NumPy
    np_fft_result = fft2(np_image)

    return np_fft_result.tolist()
    
def get2DInverseDiscreteFourierTransform(fourier_transform: DiscreteFourierTransform, image_sizes: Tuple[int, int]) -> ListImage:
    M = image_sizes[0]
    N = image_sizes[1]

    # Convert the fourier transform to a NumPy array for faster computation
    np_fourier_transform = array(fourier_transform, dtype=complex128)

    # Perform the inverse 2D FFT using NumPy
    np_inverse_fft_result = ifft2(np_fourier_transform)
    
    # Convert the result to a ListImage
    inverse_fft_result = np_inverse_fft_result.real.tolist()

    return convertToProperImage(inverse_fft_result)