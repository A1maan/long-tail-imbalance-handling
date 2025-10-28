import os
import cv2
import json
import numpy as np
import torch
from ensemble_boxes import weighted_boxes_fusion
from ultralytics import YOLO
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.transforms import functional as F
from pycocotools.coco import COCO
import sys
from datetime import datetime
from pathlib import Path

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Base detector class
class BaseDetector:
    def __init__(self, name, model, confidence_threshold=0.5):
        self.name = name
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.model.to(device)
        if hasattr(self.model, 'eval'):
            self.model.eval()
    
    def predict(self, image):
        """Return boxes, scores, labels"""
        raise NotImplementedError

# YOLOv11/YOLOv8/YOLOv12 Detector
class YOLODetector(BaseDetector):
    def predict(self, image):
        results = self.model(image, conf=self.confidence_threshold, verbose=False)
        boxes = []
        scores = []
        labels = []
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy())
                    
                    boxes.append([x1, y1, x2, y2])
                    scores.append(conf)
                    labels.append(cls)
        
        return boxes, scores, labels

# RT-DETR Detector
class RTDETRDetector(BaseDetector):
    def predict(self, image):
        results = self.model(image, conf=self.confidence_threshold, verbose=False)
        boxes = []
        scores = []
        labels = []
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy())
                    
                    boxes.append([x1, y1, x2, y2])
                    scores.append(conf)
                    labels.append(cls)
        
        return boxes, scores, labels

# Faster R-CNN v2 Detector
class FasterRCNNv2Detector(BaseDetector):
    def predict(self, image):
        # Convert BGR to RGB and normalize
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = F.to_tensor(image_rgb).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
        
        boxes = []
        scores = []
        labels = []
        
        output = outputs[0]
        pred_boxes = output['boxes'].cpu().numpy()
        pred_scores = output['scores'].cpu().numpy()
        pred_labels = output['labels'].cpu().numpy()
        
        for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
            if score >= self.confidence_threshold:
                boxes.append(box.tolist())
                scores.append(float(score))
                labels.append(int(label))
        
        return boxes, scores, labels

# Load models
print("Loading models...")
detectors = []

try:
    yolov11_model = YOLO('/home/almaankhan/model/baseline/yolo11s.pt')
    yolov11 = YOLODetector('yolov11', yolov11_model)
    detectors.append(yolov11)
    print("✓ YOLOv11 loaded")
except Exception as e:
    print(f"✗ YOLOv11 failed: {e}")

try:
    yolov8_model = YOLO('/home/almaankhan/model/baseline/yolov8s.pt')
    yolov8 = YOLODetector('yolov8', yolov8_model)
    detectors.append(yolov8)
    print("✓ YOLOv8 loaded")
except Exception as e:
    print(f"✗ YOLOv8 failed: {e}")

try:
    yolov12_model = YOLO('/home/almaankhan/model/baseline/yolo12s.pt')
    yolov12 = YOLODetector('yolov12', yolov12_model)
    detectors.append(yolov12)
    print("✓ YOLOv12 loaded")
except Exception as e:
    print(f"✗ YOLOv12 failed: {e}")

try:
    rtdetr_model = YOLO('/home/almaankhan/model/baseline/rtdetr-l.pt')
    rtdetr = RTDETRDetector('rtdetr', rtdetr_model)
    detectors.append(rtdetr)
    print("✓ RT-DETR loaded")
except Exception as e:
    print(f"✗ RT-DETR failed: {e}")

try:
    fasterrcnn_model = fasterrcnn_resnet50_fpn_v2(weights=None)
    fasterrcnn_model.load_state_dict(torch.load('/home/almaankhan/model/baseline/faster_rcnn_resnet50_fpn_v2.pt', map_location=device))
    fasterrcnn = FasterRCNNv2Detector('fasterrcnn_resnet50_fpn_v2', fasterrcnn_model)
    detectors.append(fasterrcnn)
    print("✓ Faster R-CNN v2 loaded")
except Exception as e:
    print(f"✗ Faster R-CNN v2 failed: {e}")

print(f"\n✓ {len(detectors)} models loaded successfully\n")

if len(detectors) == 0:
    print("ERROR: No models loaded!")
    sys.exit(1)

# Dataset paths - use absolute paths
BASE_DIR = Path(__file__).parent.parent.parent 
VAL_IMG_DIR = Path('/home/almaankhan/data/coco/images/val2017')
ANN_FILE = Path('/home/almaankhan/data/coco/annotations/instances_val2017.json')
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
OUTPUT_JSON = BASE_DIR / 'Results' / 'ensemble' / f'ensemble_predictions_{TIMESTAMP}.json'
OUTPUT_DIR = BASE_DIR / 'Results' / 'ensemble' / f'ensemble_predictions_{TIMESTAMP}'

# Create output directory
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Helper functions for box normalization
def normalize_boxes(boxes, w, h):
    """Normalize boxes to [0,1] range"""
    normalized = []
    for box in boxes:
        x1, y1, x2, y2 = box
        normalized.append([x1/w, y1/h, x2/w, y2/h])
    return normalized

def denormalize_boxes(boxes, w, h):
    """Denormalize boxes from [0,1] range"""
    denormalized = []
    for box in boxes:
        x1, y1, x2, y2 = box
        denormalized.append([x1*w, y1*h, x2*w, y2*h])
    return denormalized

# Load image list
print(f"Loading image list from {VAL_IMG_DIR}...")
image_files = sorted([f for f in os.listdir(str(VAL_IMG_DIR)) if f.endswith(('.jpg', '.png'))])
print(f"Found {len(image_files)} images\n")

# Run ensemble inference
print("Running ensemble inference...")
ensemble_results = []
image_id_counter = 1

for idx, img_name in enumerate(image_files):
    img_path = VAL_IMG_DIR / img_name
    
    try:
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"⚠ Failed to load image: {img_name}")
            continue
        
        h, w = image.shape[:2]
        boxes_list = []
        scores_list = []
        labels_list = []
        
        # Get predictions from all detectors
        for detector in detectors:
            try:
                boxes, scores, labels = detector.predict(image)
                if boxes:
                    # Normalize boxes for WBF
                    norm_boxes = normalize_boxes(boxes, w, h)
                    boxes_list.append(norm_boxes)
                    scores_list.append(scores)
                    labels_list.append(labels)
                else:
                    # Add empty predictions
                    boxes_list.append([])
                    scores_list.append([])
                    labels_list.append([])
            except Exception as e:
                print(f"⚠ Detector {detector.name} failed on {img_name}: {e}")
                boxes_list.append([])
                scores_list.append([])
                labels_list.append([])
        
        # Apply WBF ensemble
        try:
            ensemble_boxes, ensemble_scores, ensemble_labels = weighted_boxes_fusion(
                boxes_list, scores_list, labels_list,
                iou_thr=0.55, skip_box_thr=0.0, weights=None
            )
            # Denormalize boxes
            ensemble_boxes = denormalize_boxes(ensemble_boxes, w, h)
        except Exception as e:
            print(f"⚠ WBF failed on {img_name}: {e}")
            ensemble_boxes = []
            ensemble_scores = []
            ensemble_labels = []
        
        # Store results in COCO format
        result = {
            'image_id': image_id_counter,
            'image_name': img_name,
            'width': w,
            'height': h,
            'detections': []
        }
        
        for box, score, label in zip(ensemble_boxes, ensemble_scores, ensemble_labels):
            x1, y1, x2, y2 = box
            result['detections'].append({
                'bbox': [x1, y1, x2-x1, y2-y1],  # COCO format: [x, y, width, height]
                'score': float(score),
                'category_id': int(label)
            })
        
        ensemble_results.append(result)
        image_id_counter += 1
        
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(image_files)} images")
    
    except Exception as e:
        print(f"⚠ Error processing {img_name}: {e}")
        continue

# Save results
print(f"\nSaving results to {OUTPUT_JSON}...")
with open(str(OUTPUT_JSON), 'w') as f:
    json.dump(ensemble_results, f, indent=2)

print(f"✓ Ensemble predictions saved to {OUTPUT_JSON}")
print(f"✓ Total images processed: {len(ensemble_results)}")
print(f"✓ Output directory: {OUTPUT_DIR}")
