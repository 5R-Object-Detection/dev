import torch
import numpy as np
import math
from collections import deque
from typing import Dict, List, Tuple
from interfaces import IObjectAssociator, IPoseValidator, IMotionTracker, IPocketHandsValidator

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


class PocketHandsValidator(IPocketHandsValidator):
    """
    Validates if hands are in pockets using body-normalized kinematic variances.
    It checks two separate conditions:
    1. Elbow Swing Variance: Is the arm rigid/locked or freely swinging?
    2. Wrist Location (dy): If rigid, is the wrist located right at the hip level (pocket zone), 
       or far below the hip (holding an item)?
    """
    def __init__(self, buffer_size: int = 15):
        self.buffer_size = buffer_size
        self.history: Dict[int, Dict[str, deque]] = {}
        
    def check_hands_in_pockets(self, person_id: int, keypoints: torch.Tensor, is_walking: bool) -> bool:
        if not is_walking:
            if person_id in self.history:
                del self.history[person_id]
            return False
            
        if person_id not in self.history:
            self.history[person_id] = {
                'L_elbow': deque(maxlen=self.buffer_size),
                'L_wrist': deque(maxlen=self.buffer_size),
                'R_elbow': deque(maxlen=self.buffer_size),
                'R_wrist': deque(maxlen=self.buffer_size),
                'torso': deque(maxlen=self.buffer_size)
            }
            
        def extract_pt(idx):
            pt = keypoints[idx]
            conf = float(pt[2]) if len(pt) > 2 else 1.0
            return float(pt[0]), float(pt[1]), conf

        # Shoulders(5,6), Elbows(7,8), Wrists(9,10), Hips(11,12)
        for side, (s_idx, e_idx, w_idx, h_idx) in [('L', (5, 7, 9, 11)), ('R', (6, 8, 10, 12))]:
            sx, sy, sc = extract_pt(s_idx)
            ex, ey, ec = extract_pt(e_idx)
            wx, wy, wc = extract_pt(w_idx)
            hx, hy, hc = extract_pt(h_idx)
            
            # We need torso points to normalize scales
            if sc > 0.3 and hc > 0.3:
                torso_h = max(10.0, hy - sy) # Prevents division by zero
                self.history[person_id]['torso'].append(torso_h)
                
                # Track elbow relative to shoulder
                if ec > 0.3:
                    self.history[person_id][f'{side}_elbow'].append((ex - sx, ey - sy))
                    
                # Track wrist relative to hip
                if wc > 0.2:
                    self.history[person_id][f'{side}_wrist'].append((wx - hx, wy - hy))
                else:
                    self.history[person_id][f'{side}_wrist'].append(None)
                    
        # History check
        if len(self.history[person_id]['torso']) < self.buffer_size // 2:
            return False
            
        avg_torso = float(np.mean(self.history[person_id]['torso']))
        
        def is_side_in_pocket(side_prefix) -> bool:
            elbows = self.history[person_id][f'{side_prefix}_elbow']
            wrists = self.history[person_id][f'{side_prefix}_wrist']
            
            if len(elbows) < self.buffer_size // 2:
                return False
                
            # 1. Swing Variance Check
            elb_arr = np.array(list(elbows))
            elb_var = np.var(elb_arr[:, 0]) + np.var(elb_arr[:, 1])
            normalized_elb_var = elb_var / (avg_torso ** 2)
            
            # If variance > 0.015, arm is swinging naturally! NOT a pocket.
            if normalized_elb_var > 0.015:
                return False
                
            # 2. Hand Position Check (Holding item vs In pocket)
            valid_wrists = [w for w in wrists if w is not None]
            
            # If wrist is heavily occluded while arm is rigid -> highly likely in pocket
            if len(valid_wrists) < len(wrists) * 0.4:
                return True
                
            w_arr = np.array(valid_wrists)
            
            # Wrist Vertical Delta (dy)
            # Positive means wrist is BELOW the hip
            avg_dy = np.mean(w_arr[:, 1]) / avg_torso
            
            # Wrist Horizontal Delta (dx)
            avg_dx = np.mean(w_arr[:, 0]) / avg_torso
            
            # A pocket is horizontally close to the hip and vertically within 
            # -15% (slightly above pocket) to +25% (deep pocket) of the torso height.
            # Compare to: Carrying a bag is typically +40% to +60% straight down the thigh.
            if -0.15 <= avg_dy <= 0.25 and abs(avg_dx) < 0.4:
                return True
                
            return False

        return is_side_in_pocket('L') or is_side_in_pocket('R')
