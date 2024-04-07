import cv2
import os
import json
import numpy as np
from pathlib import Path
from scipy.interpolate import CubicSpline

from yolo_segmentation import YOLOSegmentation  # Make sure this is defined in your project

def smooth_contour_points(points, num_points=100):
    """
    Smooths contour points using cubic spline interpolation.

    Args:
    - points (list): The original contour points as a list of [x, y] pairs.
    - num_points (int): The number of points for the smoothed contour.

    Returns:
    - np.ndarray: Array of smoothed points.
    """
    # Separate x and y coordinates
    points = np.array(points)
    x = points[:, 0]
    y = points[:, 1]
    
    # Parameterize the points to a uniform domain
    t = np.linspace(0, 1, len(points))
    t_smooth = np.linspace(0, 1, num_points)
    
    # Fit cubic splines to x and y coordinates
    cs_x = CubicSpline(t, x)
    cs_y = CubicSpline(t, y)
    
    # Generate smooth x and y coordinates
    x_smooth = cs_x(t_smooth)
    y_smooth = cs_y(t_smooth)
    
    # Combine x and y coordinates
    smooth_points = np.vstack((x_smooth, y_smooth)).T
    
    return smooth_points.tolist()

def create_labelme_json(image_path, bboxes, classes, segmentations, image_shape):
    shapes = []
    for bbox, class_id, seg in zip(bboxes, classes, segmentations):
        shape = {
            "label": 'bed',
            "points": seg,  # Points are now assumed to be already smoothed
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
    ys = YOLOSegmentation("bed_pytorch_model/bed_seg_4oct23.pt")
    dataset_path = Path(dataset_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for image_file in dataset_path.glob('*.jpg'):
        img = cv2.imread(str(image_file))
        if img is None:
            continue
        bboxes, classes, segmentations, scores = ys.detect(img)
        
        for bbox, class_id, seg in zip(bboxes, classes, segmentations):
            smoothed_seg = smooth_contour_points(seg)
            cv2.polylines(img, [np.array(smoothed_seg, np.int32)], True, (0, 0, 255), 4)

        json_data = create_labelme_json(image_file, bboxes, classes, [smooth_contour_points(seg) for seg in segmentations], img.shape)
        json_path = output_path / (image_file.stem + '.json')
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)

        annotated_image_path = output_path / (image_file.stem + '_annotated.jpg')
        cv2.imwrite(str(annotated_image_path), img)

# Example usage
dataset_dir = 'dataset'
output_dir = 'output'
process_images(dataset_dir, output_dir)
