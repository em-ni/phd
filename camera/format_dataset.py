import hashlib
import json
import os
from pathlib import Path
import shutil

def convert_polygon_to_bbox(polygon):
    xs = [p['x'] for p in polygon]
    ys = [p['y'] for p in polygon]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    return (min_x, min_y, max_x - min_x, max_y - min_y)

def hash_to_int(hash_string, mod):
    return int(hashlib.sha256(hash_string.encode()).hexdigest(), 16) % mod

def process_subset(source_dir, dest_dir, labels_mapping, subset_size=None):
    # Setup directories
    images_train_dir = Path(dest_dir) / 'images' / 'train'
    images_val_dir = Path(dest_dir) / 'images' / 'val'
    labels_train_dir = Path(dest_dir) / 'labels' / 'train'
    labels_val_dir = Path(dest_dir) / 'labels' / 'val'
    directories = [images_train_dir, images_val_dir, labels_train_dir, labels_val_dir]
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

        annotation_dict = {anno['finding_id']: anno for anno in annotations}

        print(f"Processing {len(objects)} objects in {category}")

        # Process subset of objects
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
                    shutil.copy(img_file_path, new_img_path)
                    print(f"Copied {img_file_path} to {new_img_path}")

                    # Fetch and convert annotations
                    if image_id in annotation_dict:
                        annotation = annotation_dict[image_id]
                        bbox = convert_polygon_to_bbox(annotation['data'])
                        label_index = labels_mapping.get(annotation_dict[annotation['object_id']]['name'], -1)
                        if label_index == -1:
                            continue
                        label_content = f"{label_index} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n"
                        label_path = label_dir / f"{image_id}.txt"
                        with open(label_path, 'w') as f:
                            f.write(label_content)
                        print(f"Created label for {image_id} at {label_path}")

    print(f"Total images in training set: {train_count}")
    print(f"Total images in validation set: {val_count}")
    print(f"Total images: {train_count + val_count}")


if __name__ == "__main__":
    source_dir = Path(os.path.dirname(os.path.abspath(__file__))) / 'data' / 'bronchoscopy'
    dest_dir = Path(os.path.dirname(os.path.abspath(__file__))) / 'data' / 'formatted_bronchoscopy'
    labels_mapping = {
        'Vocal cords': 0,
        'Main carina': 1,
        'Mucosal infiltration': 2,
        'Mucosal edema of carina': 3,
        'Anthrocosis': 4,
        'Tumor': 5,
        'Vascular growth': 6,
        'Right superior lobar bronchus': 7,
        'Intermediate bronchus': 8,
        'Right inferior lobar bronchus': 9,
        'Right middle lobar bronchus': 10,
        'Left inferior lobar bronchus': 11,
        'Left superior lobar bronchus': 12,
        'Right main bronchus': 13,
        'Left main bronchus': 14,
        'Stenosis': 15,
        'Muscosal erythema': 16,
        'Trachea': 17
    }
    process_subset(source_dir, dest_dir, labels_mapping, subset_size=None)
