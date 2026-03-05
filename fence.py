import cv2
import numpy as np
from typing import List, Tuple
from interfaces import IFence

class VirtualFence(IFence):
    """
    Defines and manages a polygonal virtual fence.
    Responsible for geometric boundary containment checks.
    """
    def __init__(self, polygon_points: List[Tuple[int, int]]):
        self.polygon = np.array(polygon_points, np.int32)
        
    def contains(self, point: Tuple[int, int]) -> bool:
        result = cv2.pointPolygonTest(self.polygon, (float(point[0]), float(point[1])), False)
        return result >= 0
        
    def draw(self, frame: np.ndarray, color: Tuple[int, int, int], thickness: int = 2) -> None:
        cv2.polylines(frame, [self.polygon], isClosed=True, color=color, thickness=thickness)
