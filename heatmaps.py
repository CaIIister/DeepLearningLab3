import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import xml.etree.ElementTree as ET
import cv2

# Define target classes
TARGET_CLASSES = ['diningtable', 'sofa']


def generate_heatmap_visualizations(dataset_path, num_samples=3):
    # Paths to the trained models
    scratch_model_path = os.path.join(dataset_path, 'faster_rcnn_model.pth')
    transfer_model_path = os.path.join(dataset_path, 'best_faster_rcnn_model.pth')

    # Create results directory if it doesn't exist
    results_dir = os.path.join(dataset_path, 'results', 'heatmap_comparison')
    os.makedirs(results_dir, exist_ok=True)

    # Load images and annotations
    img_dir = os.path.join(dataset_path, 'images')
    ann_dir = os.path.join(dataset_path, 'annotations')

    # Get list of all image files
    img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]

    # Randomly select images
    selected_files = random.sample(img_files, num_samples)

    # Load the models
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load model trained from scratch
    scratch_model = fasterrcnn_resnet50_fpn(num_classes=len(TARGET_CLASSES) + 1)
    scratch_model.load_state_dict(torch.load(scratch_model_path, map_location=device))
    scratch_model.to(device)
    scratch_model.eval()

    # Load transfer learning model
    transfer_model = fasterrcnn_resnet50_fpn(num_classes=len(TARGET_CLASSES) + 1)
    transfer_model.load_state_dict(torch.load(transfer_model_path, map_location=device))
    transfer_model.to(device)
    transfer_model.eval()

    # Define transform for input images
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    for i, img_file in enumerate(selected_files):
        img_path = os.path.join(img_dir, img_file)
        image_id = os.path.splitext(img_file)[0]
        ann_path = os.path.join(ann_dir, f"{image_id}.xml")

        # Load image
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)

        # Load ground truth annotation
        tree = ET.parse(ann_path)
        root = tree.getroot()

        # Get ground truth boxes
        gt_boxes = []
        gt_labels = []

        for obj in root.findall('./object'):
            class_name = obj.find('name').text
            if class_name in TARGET_CLASSES:
                bbox = obj.find('bndbox')
                xmin = float(bbox.find('xmin').text)
                ymin = float(bbox.find('ymin').text)
                xmax = float(bbox.find('xmax').text)
                ymax = float(bbox.find('ymax').text)
                gt_boxes.append([xmin, ymin, xmax, ymax])
                gt_labels.append(TARGET_CLASSES.index(class_name) + 1)

        # Convert image to numpy for visualization
        img_np = np.array(img)
        height, width = img_np.shape[:2]

        # Create heatmaps
        with torch.no_grad():
            # Get predictions from both models
            scratch_preds = scratch_model(img_tensor)[0]
            transfer_preds = transfer_model(img_tensor)[0]

        # Create a figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Plot original image with ground truth
        axes[0].imshow(img_np)
        axes[0].set_title('Ground Truth')
        for box, label in zip(gt_boxes, gt_labels):
            rect = plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                 fill=False, edgecolor='green', linewidth=2)
            axes[0].add_patch(rect)
            axes[0].text(box[0], box[1], TARGET_CLASSES[label - 1],
                         bbox=dict(facecolor='green', alpha=0.5), fontsize=8)
        axes[0].axis('off')

        # Function to generate heatmap based on model predictions
        def generate_model_heatmap(predictions, ax, title, color):
            # Create a heatmap (confidence map)
            heatmap = np.zeros((height, width))

            # Extract predictions
            boxes = predictions['boxes'].cpu().numpy()
            scores = predictions['scores'].cpu().numpy()
            labels = predictions['labels'].cpu().numpy()

            # Filter predictions with confidence > 0.3
            mask = scores > 0.3
            boxes = boxes[mask]
            scores = scores[mask]
            labels = labels[mask]

            # Draw base image
            ax.imshow(img_np)

            # Add bounding boxes and create heatmap
            for box, score, label in zip(boxes, scores, labels):
                # Draw rectangle
                rect = plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                     fill=False, edgecolor=color, linewidth=2)
                ax.add_patch(rect)

                # Add confidence text
                ax.text(box[0], box[1] - 5, f"{TARGET_CLASSES[label - 1]}: {score:.2f}",
                        bbox=dict(facecolor=color, alpha=0.5), fontsize=8)

                # Add to heatmap
                x1, y1, x2, y2 = map(int, box)

                # Create gaussian blob for confidence
                y, x = np.mgrid[0:height, 0:width]
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                sigma_x = (x2 - x1) / 3  # spread based on box width
                sigma_y = (y2 - y1) / 3  # spread based on box height

                # Create gaussian
                gaussian = np.exp(-((x - center_x) ** 2 / (2 * sigma_x ** 2) +
                                    (y - center_y) ** 2 / (2 * sigma_y ** 2))) * score

                # Add to heatmap
                heatmap = np.maximum(heatmap, gaussian)

            # Normalize heatmap
            if heatmap.max() > 0:
                heatmap = heatmap / heatmap.max()

            # Apply heatmap as overlay
            heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

            # Create a blend of original image and heatmap
            alpha = 0.4  # transparency of heatmap
            overlay = img_np.copy()
            mask = heatmap > 0.1  # Only show areas with reasonable confidence
            overlay[mask] = img_np[mask] * (1 - alpha) + heatmap_colored[mask] * alpha

            # Display overlay
            ax.imshow(overlay, alpha=0.6)
            ax.set_title(title)
            ax.axis('off')

            # Return average confidence for title
            return scores.mean() if len(scores) > 0 else 0

        # Generate heatmaps for both models
        scratch_conf = generate_model_heatmap(scratch_preds, axes[1], 'From Scratch', 'red')
        transfer_conf = generate_model_heatmap(transfer_preds, axes[2], 'Transfer Learning', 'blue')

        # Update titles with average confidence
        axes[1].set_title(f'From Scratch (Avg Conf: {scratch_conf:.2f})')
        axes[2].set_title(f'Transfer Learning (Avg Conf: {transfer_conf:.2f})')

        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'heatmap_comparison_{i + 1}.png'), dpi=200)
        plt.close()

        print(f"Generated heatmap comparison {i + 1}/{num_samples}")

    print(f"All heatmap visualizations saved to {results_dir}")
    return results_dir


if __name__ == "__main__":
    generate_heatmap_visualizations('dataset_E4888')