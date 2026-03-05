import torch
import math
from collections import deque
from typing import Dict, List, Tuple
from interfaces import IObjectAssociator, IPoseValidator, IMotionTracker

class PhoneToPersonAssociator(IObjectAssociator):
    """
    Associates a phone to a person if the phone bounding box center
    is within the spatial bounds of the person's bounding box.
    """
    def associate(self, persons_boxes: torch.Tensor, phones_boxes: torch.Tensor, person_ids: List[int]) -> Dict[int, torch.Tensor]:
        associations = {}
        if len(persons_boxes) == 0 or len(phones_boxes) == 0:
            return associations
            
        for p_idx, p_box in enumerate(persons_boxes):
            p_id = person_ids[p_idx]
            px1, py1, px2, py2 = p_box
            
            for phone_box in phones_boxes:
                ph_x1, ph_y1, ph_x2, ph_y2 = phone_box
                ph_cx = (ph_x1 + ph_x2) / 2.0
                ph_cy = (ph_y1 + ph_y2) / 2.0
                
                if px1 <= ph_cx <= px2 and py1 <= ph_cy <= py2:
                    associations[p_id] = phone_box
                    break
                    
        return associations

class WristDistanceValidator(IPoseValidator):
    """
    Validates if either the left (idx 9) or right (idx 10) wrist is 
    within a threshold distance of the phone's center.
    """
    def __init__(self, threshold_pixels: float):
        self.threshold = threshold_pixels
        self.left_wrist_idx = 9
        self.right_wrist_idx = 10

    def is_pose_valid(self, person_keypoints: torch.Tensor, phone_box: torch.Tensor) -> Tuple[bool, float]:
        ph_x1, ph_y1, ph_x2, ph_y2 = phone_box
        ph_cx = float((ph_x1 + ph_x2) / 2.0)
        ph_cy = float((ph_y1 + ph_y2) / 2.0)
        
        phone_center = torch.tensor([ph_cx, ph_cy], device=person_keypoints.device)
        
        left_wrist = person_keypoints[self.left_wrist_idx][:2]
        right_wrist = person_keypoints[self.right_wrist_idx][:2]
        
        valid = False
        if left_wrist[0] > 0 and left_wrist[1] > 0:
            dist_l = torch.norm(left_wrist - phone_center)
            if dist_l <= self.threshold:
                valid = True
                
        if right_wrist[0] > 0 and right_wrist[1] > 0:
            dist_r = torch.norm(right_wrist - phone_center)
            if dist_r <= self.threshold:
                valid = True
                
        # Return distance for debugging purposes as well
        min_dist = min([d for d in [dist_l if 'dist_l' in locals() else float('inf'), 
                                    dist_r if 'dist_r' in locals() else float('inf')] if d != float('inf')] + [float('inf')])
                
        return valid, float(min_dist)

class CentroidMotionTracker(IMotionTracker):
    """
    Tracks the centroid of an ID over a bounded buffer. 
    Motion is recognized if spatial displacement exceeds a threshold.
    """
    def __init__(self, buffer_size: int, displacement_threshold: float):
        self.buffer_size = buffer_size
        self.threshold = displacement_threshold
        self.history: Dict[int, deque] = {}

    def update_and_check_motion(self, person_id: int, centroid: Tuple[float, float]) -> bool:
        if person_id not in self.history:
            self.history[person_id] = deque(maxlen=self.buffer_size)
            
        self.history[person_id].append(centroid)
        
        if len(self.history[person_id]) < self.buffer_size:
            return False
            
        oldest = self.history[person_id][0]
        newest = self.history[person_id][-1]
        
        displacement = math.hypot(newest[0] - oldest[0], newest[1] - oldest[1])
        return displacement > self.threshold
