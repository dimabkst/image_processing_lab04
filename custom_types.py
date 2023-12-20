from typing import List, Callable, Tuple, Union

ListImage = List[List[int]]

ListImageRaw = List[List[float]]

FilterKernel = List[List[float]]

PSFKernerl = List[List[float]]

FloatOrNone = Union[float, None]

ImageFunction = Callable[[int, int], int]

DiscreteFourierTransform = List[List[complex]]

DiscreteFunctionMatrix = Union[List[List[int]], List[List[float]]]

Sizes2D = Tuple[int, int]