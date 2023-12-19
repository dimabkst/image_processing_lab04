import matplotlib.pyplot as plt
from typing import List, Union
from numpy import array, clip, pad, zeros, uint8
from PIL import Image
from custom_types import ListImage, Sizes2D
from constants import MIN_INTENSITY, MAX_INTENSITY

def padMatrixWithZeros(matrix: List[List], new_sizes: Sizes2D) -> List[List]:
    original = array(matrix)

    # Calculate the padding needed on each side
    pad_height = (new_sizes[0] - len(matrix)) // 2
    pad_width = (new_sizes[1] - len(matrix[0])) // 2

    extra_padding_height = (new_sizes[0] - len(matrix)) % 2
    extra_padding_width = (new_sizes[1] - len(matrix[0])) % 2

    # Pad the original array
    new_array = pad(array=original, pad_width=((pad_height, pad_height + extra_padding_height), (pad_width, pad_width + extra_padding_width)), mode='constant', constant_values=0) # type: ignore

    return new_array.tolist()

def surroundMatrixWithZeros(matrix: List[List], new_sizes: Sizes2D) -> List[List]:
    original = array(matrix)

    # Create an array filled with zeros
    new = zeros(new_sizes, dtype=original.dtype)

    # Copy the original array to the top-left corner of the new array
    new[:original.shape[0], :original.shape[1]] = original

    return new.tolist()

def convertToProperImage(image: List[List[Union[float, complex]]]) -> ListImage:
    # Convert values to int and limit them to min_intensity, max_intensity

    # has some problem with types but still works
    return clip(a=image, a_min=MIN_INTENSITY, a_max=MAX_INTENSITY).astype(uint8).tolist() # type: ignore

def convertToListImage(image: Image.Image) -> ListImage:
    # In getpixel((x, y)) x - column, y - row 
    return [[image.getpixel((j, i)) for j in range(image.width)] for i in range(image.height)]

def convertToPillowImage(image: ListImage) -> Image.Image:
    return Image.fromarray(array(image, dtype=uint8), mode='L')

def saveImage(image: Image.Image, path: str) -> None:
    return image.save(path, mode='L')

def plot2DMatrix(matrix: List[List[float]]) -> None:
    plt.imshow(array(matrix), cmap='viridis', origin='lower')
    plt.colorbar(label='Magnitude')
    plt.title('2D DFT Magnitude Plot')
    plt.xlabel('Frequency (u)')
    plt.ylabel('Frequency (v)')
    plt.show()