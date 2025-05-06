import os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm

# Define target classes for assignment E4888
TARGET_CLASSES = ['diningtable', 'sofa']


def analyze_dataset(dataset_path='dataset_E4888'):
    """
    Analyzes the prepared dataset for the two target classes.

    Args:
        dataset_path: Path to the filtered dataset
    """
    print(f"Analyzing dataset for classes: {TARGET_CLASSES}")

    # Check if paths exist
    images_dir = os.path.join(dataset_path, 'images')
    annotations_dir = os.path.join(dataset_path, 'annotations')

    if not os.path.exists(images_dir) or not os.path.exists(annotations_dir):
        print(f"Error: Dataset not found at {dataset_path}")
        return

    # Initialize statistics
    class_counts = {cls: 0 for cls in TARGET_CLASSES}
    bbox_areas = {cls: [] for cls in TARGET_CLASSES}
    bbox_aspect_ratios = {cls: [] for cls in TARGET_CLASSES}
    image_sizes = []
    objects_per_image = []

    # Get list of annotation files
    ann_files = [f for f in os.listdir(annotations_dir) if f.endswith('.xml')]

    print("Analyzing annotations...")
    for ann_file in tqdm(ann_files):
        ann_path = os.path.join(annotations_dir, ann_file)

        # Parse XML
        tree = ET.parse(ann_path)
        root = tree.getroot()

        # Get image size
        width = int(root.find('./size/width').text)
        height = int(root.find('./size/height').text)
        image_sizes.append((width, height))

        # Count objects in this image
        objects_in_image = 0

        # Process objects
        for obj in root.findall('./object'):
            class_name = obj.find('name').text

            if class_name in TARGET_CLASSES:
                class_counts[class_name] += 1
                objects_in_image += 1

                # Extract bounding box
                bbox = obj.find('bndbox')
                xmin = float(bbox.find('xmin').text)
                ymin = float(bbox.find('ymin').text)
                xmax = float(bbox.find('xmax').text)
                ymax = float(bbox.find('ymax').text)

                # Calculate bbox properties
                width = xmax - xmin
                height = ymax - ymin
                area = width * height
                aspect_ratio = width / height if height > 0 else 0

                bbox_areas[class_name].append(area)
                bbox_aspect_ratios[class_name].append(aspect_ratio)

        objects_per_image.append(objects_in_image)

    # Print statistics
    total_images = len(ann_files)
    total_objects = sum(class_counts.values())

    print("\n===== Dataset Statistics =====")
    print(f"Total images: {total_images}")
    print(f"Total objects: {total_objects}")
    print("\nClass distribution:")
    for cls in TARGET_CLASSES:
        percentage = (class_counts[cls] / total_objects * 100) if total_objects > 0 else 0
        print(f"  - {cls}: {class_counts[cls]} instances ({percentage:.1f}%)")

    print(f"\nAverage objects per image: {np.mean(objects_per_image):.2f}")
    print(f"Max objects in a single image: {np.max(objects_per_image)}")

    # Calculate average bbox properties
    print("\nBounding box statistics:")
    for cls in TARGET_CLASSES:
        if class_counts[cls] > 0:
            avg_area = np.mean(bbox_areas[cls])
            avg_aspect = np.mean(bbox_aspect_ratios[cls])
            print(f"  - {cls}:")
            print(f"    - Average area: {avg_area:.1f} pixels²")
            print(f"    - Average aspect ratio (width/height): {avg_aspect:.2f}")

    # Create output directory for visualizations
    vis_dir = os.path.join(dataset_path, 'analysis')
    os.makedirs(vis_dir, exist_ok=True)

    # Visualize class distribution
    plt.figure(figsize=(8, 6))
    plt.bar(TARGET_CLASSES, [class_counts[cls] for cls in TARGET_CLASSES])
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    # Add count labels on top of bars
    for i, cls in enumerate(TARGET_CLASSES):
        plt.text(i, class_counts[cls] + 5, str(class_counts[cls]),
                 ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'class_distribution.png'))

    # Visualize bounding box areas
    plt.figure(figsize=(10, 6))
    for cls in TARGET_CLASSES:
        if class_counts[cls] > 0:
            plt.hist(bbox_areas[cls], alpha=0.7, label=cls, bins=20)
    plt.title('Bounding Box Area Distribution')
    plt.xlabel('Area (pixels²)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'bbox_area_distribution.png'))

    # Visualize aspect ratios
    plt.figure(figsize=(10, 6))
    for cls in TARGET_CLASSES:
        if class_counts[cls] > 0:
            plt.hist(bbox_aspect_ratios[cls], alpha=0.7, label=cls, bins=20)
    plt.title('Bounding Box Aspect Ratio Distribution')
    plt.xlabel('Aspect Ratio (width/height)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'bbox_aspect_ratio_distribution.png'))

    # Visualize objects per image
    plt.figure(figsize=(10, 6))
    plt.hist(objects_per_image, bins=range(max(objects_per_image) + 2))
    plt.title('Objects per Image')
    plt.xlabel('Number of Objects')
    plt.ylabel('Number of Images')
    plt.xticks(range(max(objects_per_image) + 1))
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'objects_per_image.png'))

    # Visualize example images with annotations
    print("\nGenerating example visualizations...")
    visualize_examples(dataset_path, vis_dir, num_examples=5)

    print(f"\nAnalysis complete! Visualizations saved to {vis_dir}")
    return class_counts, bbox_areas, bbox_aspect_ratios


def visualize_examples(dataset_path, output_dir, num_examples=5):
    """
    Creates visualizations of example images with bounding boxes.

    Args:
        dataset_path: Path to the dataset
        output_dir: Directory to save visualizations
        num_examples: Number of example images to visualize
    """
    images_dir = os.path.join(dataset_path, 'images')
    annotations_dir = os.path.join(dataset_path, 'annotations')

    # Get list of annotation files
    ann_files = [f for f in os.listdir(annotations_dir) if f.endswith('.xml')]

    # Randomly select examples
    if len(ann_files) <= num_examples:
        selected_files = ann_files
    else:
        selected_files = np.random.choice(ann_files, num_examples, replace=False)

    # Color mapping for classes
    colors = {
        'diningtable': (0, 255, 0),  # Green
        'sofa': (0, 0, 255)  # Blue
    }

    for i, ann_file in enumerate(selected_files):
        ann_path = os.path.join(annotations_dir, ann_file)
        image_id = os.path.splitext(ann_file)[0]
        img_path = os.path.join(images_dir, f"{image_id}.jpg")

        # Check if image exists
        if not os.path.exists(img_path):
            continue

        # Load image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

        # Parse XML
        tree = ET.parse(ann_path)
        root = tree.getroot()

        # Draw bounding boxes
        for obj in root.findall('./object'):
            class_name = obj.find('name').text

            if class_name in TARGET_CLASSES:
                # Extract bounding box
                bbox = obj.find('bndbox')
                xmin = int(float(bbox.find('xmin').text))
                ymin = int(float(bbox.find('ymin').text))
                xmax = int(float(bbox.find('xmax').text))
                ymax = int(float(bbox.find('ymax').text))

                # Draw rectangle
                color = colors[class_name]
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)

                # Draw label
                cv2.putText(img, class_name, (xmin, ymin - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Save visualization
        plt.figure(figsize=(10, 8))
        plt.imshow(img)
        plt.title(f'Example {i + 1}: {image_id}')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'example_{i + 1}.png'))
        plt.close()


if __name__ == "__main__":
    analyze_dataset()