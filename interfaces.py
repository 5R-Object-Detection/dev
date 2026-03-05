from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
import torch
import numpy as np

class IFence(ABC):
    """Abstract interface for region of interest boundaries."""
    @abstractmethod
    def contains(self, point: Tuple[int, int]) -> bool:
        pass
        
    @abstractmethod
    def draw(self, frame: np.ndarray, color: Tuple[int, int, int], thickness: int) -> None:
        pass

class IObjectAssociator(ABC):
    """Abstract interface for associating objects (e.g., phones to persons)."""
    @abstractmethod
    def associate(self, persons_boxes: torch.Tensor, phones_boxes: torch.Tensor, person_ids: List[int]) -> Dict[int, torch.Tensor]:
        pass

class IPoseValidator(ABC):
    """Abstract interface for validating a pose against an object."""
    @abstractmethod
    def is_pose_valid(self, person_keypoints: torch.Tensor, phone_box: torch.Tensor) -> Tuple[bool, float]:
        pass

class IMotionTracker(ABC):
    """Abstract interface for storing temporal position logic."""
    @abstractmethod
    def update_and_check_motion(self, person_id: int, centroid: Tuple[float, float]) -> bool:
        pass
