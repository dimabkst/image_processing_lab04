from typing import List
from custom_types import ListImage
from numpy import array, clip, uint8
from PIL import Image
from constants import MIN_INTENSITY, MAX_INTENSITY

def convertToProperImage(image: List[List[float]]) -> ListImage:
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