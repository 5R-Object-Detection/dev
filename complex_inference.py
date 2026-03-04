import math
import cv2
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union
import torch
import numpy as np
from collections import deque
from ultralytics import YOLO


class IFence(ABC):
    """Abstract interface for region of interest boundaries."""
    @abstractmethod
    def contains(self, point: Tuple[int, int]) -> bool:
        pass
        
    @abstractmethod
    def draw(self, frame: np.ndarray, color: Tuple[int, int, int], thickness: int) -> None:
        pass


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


class RegionSelector:
    """Handles interactive region selection using OpenCV GUI."""
    def __init__(self, source: Union[str, int]):
        self.source = source
        self.points: List[Tuple[int, int]] = []
        self.window_name = "Define Virtual Fence (Click to add points, ENTER to finish)"
        self.frame = None

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            cv2.circle(self.frame, (x, y), 5, (0, 255, 0), -1)
            if len(self.points) > 1:
                cv2.line(self.frame, self.points[-2], self.points[-1], (0, 255, 0), 2)
            cv2.imshow(self.window_name, self.frame)

    def select_region(self) -> List[Tuple[int, int]]:
        cap = cv2.VideoCapture(self.source)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise RuntimeError(f"Failed to read first frame from {self.source}")
            
        self.frame = frame.copy()
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        
        cv2.imshow(self.window_name, self.frame)
        print("Click on the image to select polygon vertices.")
        print("Press ENTER or SPACE when finished.")
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 13 or key == 32:  # Enter or Space
                break
            elif key == 27:  # Esc
                self.points = []
                break
                
        cv2.destroyWindow(self.window_name)
        
        if len(self.points) < 3:
            print("Warning: Polygon needs at least 3 points. Using default rectangle.")
            h, w = frame.shape[:2]
            return [(10, 10), (w-10, 10), (w-10, h-10), (10, h-10)]
            
        return self.points



class IObjectAssociator(ABC):
    """Abstract interface for associating objects (e.g., phones to persons)."""
    @abstractmethod
    def associate(self, persons_boxes: torch.Tensor, phones_boxes: torch.Tensor, person_ids: List[int]) -> Dict[int, torch.Tensor]:
        pass


class IPoseValidator(ABC):
    """Abstract interface for validating a pose against an object."""
    @abstractmethod
    def is_pose_valid(self, person_keypoints: torch.Tensor, phone_box: torch.Tensor) -> bool:
        pass


class IMotionTracker(ABC):
    """Abstract interface for storing temporal position logic."""
    @abstractmethod
    def update_and_check_motion(self, person_id: int, centroid: Tuple[float, float]) -> bool:
        pass


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

    def is_pose_valid(self, person_keypoints: torch.Tensor, phone_box: torch.Tensor) -> bool:
        ph_x1, ph_y1, ph_x2, ph_y2 = phone_box
        ph_cx = float((ph_x1 + ph_x2) / 2.0)
        ph_cy = float((ph_y1 + ph_y2) / 2.0)
        
        # Move CPU conversion here if tensor sizes are small to avoid device issues
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
                
        return valid, min_dist


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
        annotated_frame = p_result.plot(img=annotated_frame)
        
        # Extract Phones
        phone_boxes = torch.empty((0, 4), device=self.device)
        if detect_results and detect_results[0].boxes is not None:
            d_result = detect_results[0]
            phone_boxes = d_result.boxes.xyxy
            # Overlay phone annotations
            annotated_frame = d_result.plot(img=annotated_frame)
        
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

if __name__ == "__main__":
    print("Select Video Source:")
    print("1. Internal Camera")
    print("2. Local Video File")
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        video_source = 1  # Default internal camera
    else:
        video_source = "/Users/akmalramadhan76gmail.com/Local Documents/Toyota/5R/sample_video/sample_video_1.mp4"
    
    # Interactive polygon selection for virtual fence
    selector = RegionSelector(source=video_source)
    fence_points = selector.select_region()
    fence_instance = VirtualFence(polygon_points=fence_points)

    associator = PhoneToPersonAssociator()
    # Assuming wrist threshold of 100px (increased for better webcam tolerance)
    validator = WristDistanceValidator(threshold_pixels=100.0)
    # Assumes a displacement of >10px over 15 frames is walking (decreased for indoor walking)
    tracker = CentroidMotionTracker(buffer_size=15, displacement_threshold=10.0)
    
    analyzer = BehavioralAnalyzer(
        pose_model_path="yolo11n-pose.pt", 
        detect_model_path="yolo11n.pt", 
        associator=associator,
        validator=validator,
        tracker=tracker,
        fences=[fence_instance]
    )
    
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Failed to load video source: {video_source}")
        exit()
        
    prev_t = time.time()
        
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        alert_ids, fence_violator_ids, annotated_frame, debug_distances = analyzer.process_frame(frame)
        
        curr_t = time.time()
        fps = 1 / (curr_t - prev_t) if curr_t > prev_t else 0
        prev_t = curr_t
        
        if alert_ids:
            cv2.putText(annotated_frame, f"PHONE USAGE IN MOTION (ID: {alert_ids})", 
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)  # Orange Alert
                        
        if fence_violator_ids:
            cv2.putText(annotated_frame, f"FENCE VIOLATION (ID: {fence_violator_ids})", 
                        (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)  # Red Alert
                        
        # Display debug info to see what the system calculates
        for p_id, dist in debug_distances.items():
            if dist != float('inf'):
                cv2.putText(annotated_frame, f"ID {p_id} Wrist-Phone Dist: {dist:.1f}px", 
                            (20, 140 + (p_id*30)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                        
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_frame, "Press 'q' or ESC to quit", (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow("Dashboard: Complex Behavioral Inference + Virtual Fence", annotated_frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q'), ord('Q')):
            break
            
    cap.release()
    cv2.destroyAllWindows()
