import cv2
import torch
import numpy as np
from typing import List, Tuple, Dict
from ultralytics import YOLO
from interfaces import IFence, IObjectAssociator, IPoseValidator, IMotionTracker, IPocketHandsValidator


class BehavioralAnalyzer:
    """
    Orchestrates tracking, object association, pose validation, 
    and motion analysis to detect complex behavioral states.
    Uses dual-model inference: Pose model for people, Detection model for phones.
    """
    def __init__(self, 
                 pose_model_path: str, 
                 detect_model_path: str,
                 associator: IObjectAssociator, 
                 validator: IPoseValidator, 
                 tracker: IMotionTracker,
                 pocket_v: IPocketHandsValidator = None,
                 fences: List[IFence] = None,
                 person_class_id: int = 0,
                 phone_class_id: int = 67,
                 phone_conf_threshold: float = 0.15,
                 phone_imgsz: int = 640):
        
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
        self.pocket_validator = pocket_v
        self.fences = fences if fences else []
        self.person_class_id = person_class_id
        self.phone_class_id = phone_class_id
        self.phone_conf = phone_conf_threshold
        self.phone_imgsz = phone_imgsz

    def process_frame(self, frame: np.ndarray, is_image: bool = False, 
                      enable_fence: bool = True, enable_phone: bool = True, enable_pocket: bool = True) -> Tuple[List[int], List[int], List[int], np.ndarray, Dict[int, float]]:
        use_half = self.device != "cpu"
        
        # 1. Pose Model Inference (for tracking people and keypoints)
        pose_results = self.pose_model.track(
            frame, 
            persist=True, 
            verbose=False, 
            device=self.device, 
            half=use_half, 
            tracker="botsort.yaml",
            stream=False
        )
        
        # 2. Object Detection Inference (for detecting phones)
        # Higher imgsz and lower conf for small phone detection
        detect_results = None
        if enable_phone:
            detect_results = self.detect_model(
                frame, 
                verbose=False, 
                device=self.device, 
                half=use_half,
                classes=[self.phone_class_id],
                conf=self.phone_conf,
                imgsz=self.phone_imgsz
            )
        
        annotated_frame = frame.copy()
        
        if enable_fence:
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
        
        # Render pose annotations with compact styling
        annotated_frame = p_result.plot(img=annotated_frame, line_width=1, font_size=0.5)
        
        # Extract Phones
        phone_boxes = torch.empty((0, 4), device=self.device)
        if detect_results and detect_results[0].boxes is not None and len(detect_results[0].boxes) > 0:
            d_result = detect_results[0]
            phone_boxes = d_result.boxes.xyxy
            
            # Custom compact phone annotation
            for i, box in enumerate(d_result.boxes.xyxy.cpu().numpy().astype(int)):
                x1, y1, x2, y2 = box
                conf = float(d_result.boxes.conf[i])
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
                label = f"Phone {conf:.0%}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(annotated_frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 200, 255), -1)
                cv2.putText(annotated_frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        
        associations = self.associator.associate(person_boxes, phone_boxes, person_ids)
        alert_ids = []
        fence_violator_ids = []
        pocket_violator_ids = []
        debug_distances = {}
        
        for p_idx, p_id in enumerate(person_ids):
            p_box = person_boxes[p_idx]
            centroid = (float((p_box[0] + p_box[2]) / 2.0), float((p_box[1] + p_box[3]) / 2.0))
            
            # Fence check uses bottom-center of bounding box (feet)
            feet_coord = ((p_box[0] + p_box[2]) / 2.0, p_box[3])
            
            if enable_fence:
                for fence in self.fences:
                    if fence.contains(feet_coord):
                        fence_violator_ids.append(p_id)
                        fence.draw(annotated_frame, color=(0, 0, 255), thickness=3)
                        break
            
            # For single images, bypass the motion requirement since there is no temporal movement
            is_walking = True if is_image else self.motion_tracker.update_and_check_motion(p_id, centroid)
            
            if enable_phone and p_id in associations:
                phone_box = associations[p_id]
                is_holding, dist = self.validator.is_pose_valid(person_kpts[p_idx], phone_box)
                debug_distances[p_id] = dist
                
                if is_walking and is_holding:
                    alert_ids.append(p_id)
                    
            if enable_pocket and self.pocket_validator is not None:
                is_hands_pocket = self.pocket_validator.check_hands_in_pockets(p_id, person_kpts[p_idx], is_walking)
                if is_hands_pocket:
                    pocket_violator_ids.append(p_id)
                    
        return alert_ids, fence_violator_ids, pocket_violator_ids, annotated_frame, debug_distances
