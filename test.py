"""
Human Detection and Area Monitoring System

This module provides a production-ready, hardware-accelerated computer vision
pipeline for human detection and region-of-interest monitoring. It follows SOLID
principles, utilizing dependency injection for extensibility.
"""

import cv2
import numpy as np
import torch
import time
from abc import ABC, abstractmethod
from typing import List, Tuple
from ultralytics import YOLO


class RegionSelector:
    """Handles interactive region selection using OpenCV GUI."""
    def __init__(self, source: str):
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


class IDetector(ABC):
    """Abstract interface for object detection models."""
    @abstractmethod
    def detect_humans(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        pass


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


class YoloDetector(IDetector):
    """
    Encapsulates the YOLO object detection model.
    Manages hardware acceleration placement and inference execution.
    """
    def __init__(self, model_path: str):
        if torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        self.model = YOLO(model_path)
        self.model.to(self.device)
        
    def detect_humans(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        use_half = self.device != "cpu"
        results = self.model(frame, classes=[0], verbose=False, device=self.device, half=use_half)
        bboxes = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                bboxes.append((x1, y1, x2, y2))
        return bboxes


class VideoProcessor:
    """
    Orchestrates the video processing pipeline.
    Utilizes dependency injection for the detector and virtual fences to ensure
    modularity and adherence to the Dependency Inversion Principle.
    """
    def __init__(self, detector: IDetector, fences: List[IFence], source: str):
        self.detector = detector
        self.fences = fences
        self.source = source
        
    def _get_bottom_center(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, y2)
        
    def process(self) -> None:
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {self.source}")
            
        prev_time = time.time()
            
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            bboxes = self.detector.detect_humans(frame)
            human_count = len(bboxes)
            alert_active = False
            
            for fence in self.fences:
                fence.draw(frame, color=(255, 0, 0), thickness=2)
                
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox
                feet_coord = self._get_bottom_center(bbox)
                
                person_in_restricted_area = False
                for fence in self.fences:
                    if fence.contains(feet_coord):
                        person_in_restricted_area = True
                        alert_active = True
                        fence.draw(frame, color=(0, 0, 255), thickness=3)
                        break
                
                bbox_color = (0, 0, 255) if person_in_restricted_area else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, 2)
                cv2.circle(frame, feet_coord, 5, bbox_color, -1)
                
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if curr_time > prev_time else 0
            prev_time = curr_time
                
            status_text = f"Count: {human_count} | Alert: {'ACTIVE' if alert_active else 'NONE'} | FPS: {fps:.1f}"
            status_color = (0, 0, 255) if alert_active else (0, 255, 0)
            cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)
            cv2.putText(frame, "Press 'q' or ESC to quit", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.imshow("Monitor Dashboard", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                break
                
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    video_file = "/Users/akmalramadhan76gmail.com/Local Documents/Toyota/5R/sample_video/sample_video_2.mp4"
    
    # Interactive polygon selection
    selector = RegionSelector(source=video_file)
    fence_points = selector.select_region()
    
    detector_instance = YoloDetector(model_path="yolov26x.pt")
    fence_instance = VirtualFence(polygon_points=fence_points)
    
    processor = VideoProcessor(
        detector=detector_instance,
        fences=[fence_instance],
        source=video_file
    )
    
    try:
        processor.process()
    except Exception as e:
        print(f"Execution Error: {str(e)}")
