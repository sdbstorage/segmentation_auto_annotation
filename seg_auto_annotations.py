import cv2
import os
import json
from pathlib import Path
import cv2
from yolo_segmentation import YOLOSegmentation
import numpy as np


def create_labelme_json(image_path, bboxes, classes, segmentations, image_shape):
    shapes = []
    for bbox, class_id, seg in zip(bboxes, classes, segmentations):
        shape = {
            # "label": str(class_id),
            "label": 'bed',
            "points": seg.tolist(), # Assuming seg is a numpy array
            "group_id": None,
            "shape_type": "polygon",
            "flags": {}
        }
        shapes.append(shape)
    
    data = {
        "version": "4.5.6",
        "flags": {},
        "shapes": shapes,
        "imagePath": os.path.basename(image_path),
        "imageData": None,
        "imageHeight": image_shape[0],
        "imageWidth": image_shape[1]
    }
    return data

def process_images(dataset_dir, output_dir):
    ys = YOLOSegmentation("bed_pytorch_model/bed_seg_4oct23.pt") # Load your model
    dataset_path = Path(dataset_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for image_file in dataset_path.glob('*.jpg'): # Adjust the glob pattern based on your dataset
        img = cv2.imread(str(image_file))
        if img is None:
            continue
        bboxes, classes, segmentations, scores = ys.detect(img)
        
        # Draw bounding boxes and segmentation polylines on the image (optional)
        for bbox, class_id, seg in zip(bboxes, classes, segmentations):
            (x, y, x2, y2) = bbox
            cv2.rectangle(img, (x, y), (x2, y2), (255, 0, 0), 2)
            cv2.polylines(img, [np.array(seg)], True, (0, 0, 255), 4)

        json_data = create_labelme_json(image_file, bboxes, classes, segmentations, img.shape)
        json_path = output_path / (image_file.stem + '.json')
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)

        # annotated_image_path = output_path / (image_file.stem + '_annotated.jpg')
        annotated_image_path = output_path / (image_file.stem + '.jpg')
        cv2.imwrite(str(annotated_image_path), img, [cv2.IMWRITE_JPEG_QUALITY, 50])

# Example usage
dataset_dir = 'filtered_fall_images'
output_dir = 'output'
process_images(dataset_dir, output_dir)