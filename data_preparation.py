import os
import shutil
import xml.etree.ElementTree as ET
from tqdm import tqdm
import torch
from torchvision.datasets import VOCDetection
from PIL import Image

# Define target classes for assignment E4888
TARGET_CLASSES = ['diningtable', 'sofa']


def prepare_dataset(output_path='dataset_E4888', year='2012'):
    """
    Automatically downloads VOC dataset and prepares a filtered dataset
    containing only images with diningtable and sofa classes.

    Args:
        output_path: Output directory for the filtered dataset
        year: VOC dataset year ('2007' or '2012')
    """
    print(f"Downloading and preparing dataset for classes: {TARGET_CLASSES}")

    # Create directories
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'annotations'), exist_ok=True)

    # Download VOC dataset using torchvision
    print(f"Downloading VOC{year} dataset (if not already downloaded)...")
    voc_dataset = VOCDetection(root='./data', year=year, image_set='trainval', download=True)

    # Count for statistics
    total_images = 0
    class_counts = {cls: 0 for cls in TARGET_CLASSES}

    # Process dataset
    print("Filtering dataset...")
    for idx in tqdm(range(len(voc_dataset))):
        try:
            # Get image and annotation
            img, annotation = voc_dataset[idx]

            # Check if annotation contains target classes
            has_target_class = False
            class_in_image = {cls: 0 for cls in TARGET_CLASSES}

            for obj in annotation['annotation']['object']:
                class_name = obj['name']
                if class_name in TARGET_CLASSES:
                    has_target_class = True
                    class_in_image[class_name] += 1

            # If image contains at least one target class, save it
            if has_target_class:
                # Get image ID
                image_id = annotation['annotation']['filename'].split('.')[0]

                # Save image
                img_path = os.path.join(output_path, 'images', f"{image_id}.jpg")
                img.save(img_path)

                # Save annotation as XML
                create_xml_annotation(annotation, os.path.join(output_path, 'annotations', f"{image_id}.xml"))

                # Update statistics
                total_images += 1
                for cls in TARGET_CLASSES:
                    class_counts[cls] += class_in_image[cls]
        except Exception as e:
            print(f"Error processing image {idx}: {e}")
            continue

    # Print statistics
    print("\nDataset preparation complete!")
    print(f"Total images: {total_images}")
    for cls, count in class_counts.items():
        print(f"Class '{cls}': {count} instances")

    return total_images, class_counts


def create_xml_annotation(annotation_dict, output_path):
    """
    Converts annotation dictionary from VOCDetection to XML format.
    Only includes the target classes.

    Args:
        annotation_dict: Dictionary containing annotation information
        output_path: Path to save the XML file
    """
    annotation = annotation_dict['annotation']

    # Create root element
    root = ET.Element('annotation')

    # Add basic info
    ET.SubElement(root, 'folder').text = 'VOC2012'
    ET.SubElement(root, 'filename').text = annotation['filename']

    # Add size information
    size = ET.SubElement(root, 'size')
    ET.SubElement(size, 'width').text = str(annotation['size']['width'])
    ET.SubElement(size, 'height').text = str(annotation['size']['height'])
    ET.SubElement(size, 'depth').text = str(annotation['size']['depth'])

    # Add object information (only target classes)
    for obj in annotation['object']:
        if obj['name'] in TARGET_CLASSES:
            obj_elem = ET.SubElement(root, 'object')
            ET.SubElement(obj_elem, 'name').text = obj['name']
            ET.SubElement(obj_elem, 'pose').text = obj['pose']
            ET.SubElement(obj_elem, 'truncated').text = str(obj['truncated'])
            ET.SubElement(obj_elem, 'difficult').text = str(obj['difficult'])

            bbox = ET.SubElement(obj_elem, 'bndbox')
            ET.SubElement(bbox, 'xmin').text = str(obj['bndbox']['xmin'])
            ET.SubElement(bbox, 'ymin').text = str(obj['bndbox']['ymin'])
            ET.SubElement(bbox, 'xmax').text = str(obj['bndbox']['xmax'])
            ET.SubElement(bbox, 'ymax').text = str(obj['bndbox']['ymax'])

    # Create XML tree and save
    tree = ET.ElementTree(root)
    tree.write(output_path)


if __name__ == "__main__":
    prepare_dataset()