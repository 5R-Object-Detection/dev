import os
import glob
from pathlib import Path
from typing import List
import torch
from ultralytics import YOLO

class IAnnotationMerger:
    """Abstract interface for merging annotations."""
    def merge(self, existing_labels: List[str], new_labels: List[str]) -> List[str]:
        raise NotImplementedError

class YOLOAnnotationMerger(IAnnotationMerger):
    """Merges new YOLO predictions with existing text labels by appending."""
    def merge(self, existing_labels: List[str], new_labels: List[str]) -> List[str]:
        return existing_labels + new_labels

class PseudoLabeler:
    """
    Auto-annotates missing person bounding boxes and keypoints, merging them
    with existing bounding box annotations.
    """
    def __init__(self, model_path: str, merger: IAnnotationMerger):
        if torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
            
        self.model = YOLO(model_path)
        self.model.to(self.device)
        self.merger = merger

    def _read_existing_labels(self, label_path: Path) -> List[str]:
        if not label_path.exists():
            return []
        with open(label_path, "r") as f:
            return [line.strip() for line in f.readlines() if line.strip()]

    def _write_labels(self, label_path: Path, labels: List[str]) -> None:
        with open(label_path, "w") as f:
            f.write("\n".join(labels))

    def _extract_pose_annotations(self, result) -> List[str]:
        annotations = []
        if result.boxes is None or result.keypoints is None:
            return annotations

        boxes = result.boxes.xywhn.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        keypoints = result.keypoints.xyn.cpu().numpy()

        for idx, (box, cls, kpts) in enumerate(zip(boxes, classes, keypoints)):
            if int(cls) == 0:  # Assuming 0 is the person class
                x_c, y_c, w, h = box
                kpt_str = " ".join([f"{kp[0]:.6f} {kp[1]:.6f} 2.0" for kp in kpts])
                ann = f"0 {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f} {kpt_str}"
                annotations.append(ann)
        return annotations

    def process_dataset(self, images_dir: str, labels_dir: str, output_labels_dir: str) -> None:
        img_paths = glob.glob(os.path.join(images_dir, "*.*"))
        Path(output_labels_dir).mkdir(parents=True, exist_ok=True)

        use_half = self.device != "cpu"

        for img_path in img_paths:
            img_name = Path(img_path).stem
            label_path = Path(labels_dir) / f"{img_name}.txt"
            out_label_path = Path(output_labels_dir) / f"{img_name}.txt"

            existing_labels = self._read_existing_labels(label_path)
            
            results = self.model(img_path, verbose=False, device=self.device, half=use_half)
            
            new_labels = self._extract_pose_annotations(results[0])
            merged_labels = self.merger.merge(existing_labels, new_labels)
            
            self._write_labels(out_label_path, merged_labels)

if __name__ == "__main__":
    annotator = PseudoLabeler(model_path="yolo11x-pose.pt", merger=YOLOAnnotationMerger())
    # Example Usage:
    # annotator.process_dataset("data/images", "data/labels", "data/merged_labels")
