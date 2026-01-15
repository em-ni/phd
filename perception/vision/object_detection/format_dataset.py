import hashlib
import json
import os
from pathlib import Path
import shutil
import cv2
from ultralytics.data.utils import compress_one_image
from ultralytics.utils.downloads import zip_directory

def get_bounding_box(polygon_points, image_width, image_height):
    # Initialize min and max coordinates with the first point values
    min_x = max_x = polygon_points[0]['x']
    min_y = max_y = polygon_points[0]['y']
    
    # Loop through all points to find the min and max of x and y coordinates
    for point in polygon_points:
        min_x = min(min_x, point['x'])
        max_x = max(max_x, point['x'])
        min_y = min(min_y, point['y'])
        max_y = max(max_y, point['y'])
    
    # The bounding box is given by the top-left and bottom-right corners
    bounding_box = [min_x, min_y, max_x - min_x, max_y - min_y]
    # Normalize the bounding box coordinates
    bounding_box = [bounding_box[0] / image_width, bounding_box[1] / image_height, bounding_box[2] / image_width, bounding_box[3] / image_height]
    return bounding_box

def draw_bbox(image, bbox):
    x, y, w, h = bbox
    # Convert to integers
    x, y, w, h = map(int, [x, y, w, h])
    # Check dimensions
    if x + w > image.shape[1] or y + h > image.shape[0]:
        print(f"Bounding box {bbox} is out of image bounds.")
        return
    # Draw the rectangle
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)


def draw_circle(image, point):
    x, y = point['x'], point['y']
    # Convert to integers
    x, y = int(x), int(y)
    # Check if the point is within the image boundaries
    if x < 0 or y < 0 or x >= image.shape[1] or y >= image.shape[0]:
        print(f"Point ({x}, {y}) is out of image bounds.")
        return
    # Draw the circle
    cv2.circle(image, (x, y), 2, (0, 0, 255), -1)

def hash_to_int(hash_string, mod):
    return int(hashlib.sha256(hash_string.encode()).hexdigest(), 16) % mod

def load_labels_mapping(labels_json_path):
    with open(labels_json_path) as f:
        labels = json.load(f)
    return {label['id']: {'yolo_id': label.get('yolo_id', -1)} for label in labels}


def process_subset(source_dir, dest_dir, labels_mapping_path, subset_size=None):
    labels_mapping = load_labels_mapping(labels_mapping_path)
    
    # Setup directories
    images_train_dir = Path(dest_dir) / 'images' / 'train'
    images_val_dir = Path(dest_dir) / 'images' / 'val'
    labels_train_dir = Path(dest_dir) / 'labels' / 'train'
    labels_val_dir = Path(dest_dir) / 'labels' / 'val'
    visualization_dir = Path(dest_dir) / 'visualization'
    directories = [images_train_dir, images_val_dir, labels_train_dir, labels_val_dir, visualization_dir]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    # Counter for number of images in train and val
    train_count = 0
    val_count = 0
    
    # Process each category directory
    for category in ['Lung_cancer/Lung_cancer', 'Non_lung_cancer/Non_lung_cancer']:

        imgs_path = Path(source_dir) / category / 'imgs'
        objects_file = Path(source_dir) / category / 'objects.json'
        annotation_file = Path(source_dir) / category / 'annotation.json'
        
        with open(objects_file) as f:
            objects = json.load(f)
        with open(annotation_file) as f:
            annotations = json.load(f)

        print(f"Processing {len(objects)} objects in {category}")

        # Process images
        for obj in objects if subset_size is None else objects[:subset_size]:
            for video in obj['videos']:
                video_id = video['video_id']
                for image in video['images']:
                    image_id = image['image_id']
                    img_file_path = imgs_path / obj['id'] / video_id / f"{image_id}.png"
                    if not img_file_path.exists():
                        print(f"Image file does not exist: {img_file_path}")
                        continue  # Skip if file does not exist
                    
                    # 20% of images go to validation set
                    if hash_to_int(image_id, 5) > 0:
                        img_dir = images_train_dir
                        label_dir = labels_train_dir
                        train_count += 1
                    else:
                        img_dir = images_val_dir
                        label_dir = labels_val_dir
                        val_count += 1

                    # Copy image to new location
                    new_img_path = img_dir / f"{image_id}.png"
                    # Compress the image at its current location before moving it
                    compress_one_image(img_file_path)
                    shutil.copy(img_file_path, new_img_path)
                    print(f"Copied {img_file_path} to {new_img_path}")

        # Process annotations and create YOLO labels
        for annotation in annotations:
            file_name = annotation['object_id']
            label_id = annotation['label_ids'][0]
            polygon_points = annotation['data']
            img_path = images_train_dir / f"{file_name}.png" if (images_train_dir / f"{file_name}.png").exists() else images_val_dir / f"{file_name}.png"
            image = cv2.imread(str(img_path))
            # Get image dimensions
            image_height, image_width, _ = image.shape
            print(f"Processing annotation for {file_name}")
            print(f"Image dimensions: {image_width}x{image_height}")
            print(f"Labels: {label_id}")
            # print(f"Polygon: {polygon_points}")
            file_content = ""
            label = labels_mapping.get(label_id, None)
            if label is None:
                print(f"Label ID {label_id} not found in labels mapping")
                continue
            yolo_id = label['yolo_id']
            if yolo_id == -1:
                print(f"YOLO ID not found for label ID {label_id}")
                continue
            print(f"YOLO ID: {yolo_id}")
            
            # Get bounding boxes
            bbox = get_bounding_box(polygon_points, image_width, image_height)
            file_content += f"{yolo_id} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n"

            # Visualize bounding box
            draw_bbox(image, bbox)

            # Visualize the polygon
            for point in polygon_points:
                draw_circle(image, point)
            visualization_path = visualization_dir / f"{file_name}.png"
            cv2.imwrite(str(visualization_path), image)

            if file_content:
                if hash_to_int(file_name, 5) > 0:
                    label_dir = labels_train_dir
                else:
                    label_dir = labels_val_dir
                label_file_path = label_dir / f"{file_name}.txt"
                with open(label_file_path, 'w') as f:
                    f.write(file_content)
                print(f"Created label file {label_file_path}")

    print(f"Total images in training set: {train_count}")
    print(f"Total images in validation set: {val_count}")
    print(f"Total images: {train_count + val_count}")

    # Zip the directories
    zip_directory(dest_dir)
    

if __name__ == "__main__":
    source_dir = Path(os.path.dirname(os.path.abspath(__file__))) / 'data' / 'bronchoscopy'
    dest_dir = Path(os.path.dirname(os.path.abspath(__file__))) / 'data' / 'formatted_bronchoscopy'
    labels_mapping_path = Path(os.path.dirname(os.path.abspath(__file__))) / 'data' / 'formatted_bronchoscopy' / 'yolo_labels.json'
    process_subset(source_dir, dest_dir, labels_mapping_path, subset_size=None)
