import numpy as np

from typing import List
from vamr_types import PixelCoords

def findFundementalMat(
    points1:List[PixelCoords],
    points2:List[PixelCoords],
    method:str,
    ransacReprojThreshold:float,
    confidence:float
):
    pass