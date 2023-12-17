from typing import List, Callable, Union

ListImage = List[List[int]]

ListImageRaw = List[List[float]]

FilterKernel = List[List[float]]

FloatOrNone = Union[float, None]

ImageFunction = Callable[[int, int], int]
