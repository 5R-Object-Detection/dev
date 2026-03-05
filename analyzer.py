import torch
import numpy as np
from typing import List, Tuple, Dict
from ultralytics import YOLO
from interfaces import IFence, IObjectAssociator, IPoseValidator, IMotionTracker


class BehavioralAnalyzer:
    """
    Orchestrates tracking, object association, pose validation, 
    and motion analysis to detect complex behavioral states.
    """
    def __init__(self, 
                 pose_model_path: str, 
                 detect_model_path: str,
                 associator: IObjectAssociator, 
                 validator: IPoseValidator, 
                 tracker: IMotionTracker,
                 fences: List[IFence] = None,
                 person_class_id: int = 0,
                 phone_class_id: int = 67):
        
        if torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
            
        self.pose_model = YOLO(pose_model_path)
        self.pose_model.to(self.device)
        
        self.detect_model = YOLO(detect_model_path)
        self.detect_model.to(self.device)
        
        self.associator = associator
        self.validator = validator
        self.motion_tracker = tracker
        self.fences = fences if fences else []
        self.person_class_id = person_class_id
        self.phone_class_id = phone_class_id

    def process_frame(self, frame: np.ndarray) -> Tuple[List[int], List[int], np.ndarray, Dict[int, float]]:
        use_half = self.device != "cpu"
        
        # 1. Pose Model Inference (for tracking people and keypoints)
        pose_results = self.pose_model.track(
            frame, 
            persist=True, 
            verbose=False, 
            device=self.device, 
            half=use_half, 
            tracker="botsort.yaml"
        )
        
        # 2. Object Detection Inference (for detecting phones)
        detect_results = self.detect_model(
            frame, 
            verbose=False, 
            device=self.device, 
            half=use_half,
            classes=[self.phone_class_id]
        )
        
        annotated_frame = frame.copy()
        
        for fence in self.fences:
            fence.draw(annotated_frame, color=(255, 0, 0), thickness=2)
            
        if not pose_results or pose_results[0].boxes is None or pose_results[0].boxes.id is None:
            return [], [], annotated_frame, {}
            
        # Extract People
        p_result = pose_results[0]
        person_boxes = p_result.boxes.xyxy
        person_ids = p_result.boxes.id.int().tolist()
        
        if p_result.keypoints is None:
            return [], [], annotated_frame, {}
            
        person_kpts = p_result.keypoints.xy
        
        # Plotting pose annotations manually or via plotting tool
        annotated_frame = p_result.plot(img=annotated_frame, line_width=1, font_size=1)
        
        # Extract Phones
        phone_boxes = torch.empty((0, 4), device=self.device)
        if detect_results and detect_results[0].boxes is not None:
            d_result = detect_results[0]
            phone_boxes = d_result.boxes.xyxy
            # Overlay phone annotations
            annotated_frame = d_result.plot(img=annotated_frame, line_width=1, font_size=1)
        
        associations = self.associator.associate(person_boxes, phone_boxes, person_ids)
        alert_ids = []
        fence_violator_ids = []
        
        # Debugging dictionary to display distances on screen
        debug_distances = {}
        
        for p_idx, p_id in enumerate(person_ids):
            p_box = person_boxes[p_idx]
            centroid = (float((p_box[0] + p_box[2]) / 2.0), float((p_box[1] + p_box[3]) / 2.0))
            
            # Fence check uses bottom-center of bounding box (feet)
            feet_coord = ((p_box[0] + p_box[2]) / 2.0, p_box[3])
            
            for fence in self.fences:
                if fence.contains(feet_coord):
                    fence_violator_ids.append(p_id)
                    fence.draw(annotated_frame, color=(0, 0, 255), thickness=3)
                    break
            
            is_walking = self.motion_tracker.update_and_check_motion(p_id, centroid)
            
            if p_id in associations:
                phone_box = associations[p_id]
                is_holding, dist = self.validator.is_pose_valid(person_kpts[p_idx], phone_box)
                
                # Add to debug dict
                debug_distances[p_id] = dist
                
                # Check ALL required conditions (Association AND Pose AND Walking)
                if is_walking and is_holding:
                    alert_ids.append(p_id)
                    
        return alert_ids, fence_violator_ids, annotated_frame, debug_distances
