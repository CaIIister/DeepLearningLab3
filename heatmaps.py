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


def generate_heatmap_visualizations(dataset_path, num_samples=5):
    """
    Generate heatmap visualizations comparing models trained from scratch vs transfer learning.
    Only shows heatmaps without bounding boxes.
    """
    # Paths to the trained models
    scratch_model_path = os.path.join(dataset_path, 'best_scratch_model.pth')
    transfer_model_path = os.path.join(dataset_path, 'best_transfer_model.pth')

    # Check if models exist
    if not os.path.exists(scratch_model_path):
        print(f"Warning: Scratch model not found at {scratch_model_path}")
        scratch_model_path = os.path.join(dataset_path, 'final_scratch_model.pth')
        if not os.path.exists(scratch_model_path):
            raise FileNotFoundError(f"No scratch model found!")

    if not os.path.exists(transfer_model_path):
        print(f"Warning: Transfer model not found at {transfer_model_path}")
        transfer_model_path = os.path.join(dataset_path, 'final_transfer_model.pth')
        if not os.path.exists(transfer_model_path):
            raise FileNotFoundError(f"No transfer learning model found!")

    print(f"Using model files:")
    print(f"  Scratch model: {scratch_model_path}")
    print(f"  Transfer model: {transfer_model_path}")

    # Create results directory
    results_dir = os.path.join(dataset_path, 'results', 'heatmap_comparison')
    os.makedirs(results_dir, exist_ok=True)

    # Load images and annotations
    img_dir = os.path.join(dataset_path, 'images')
    ann_dir = os.path.join(dataset_path, 'annotations')

    # Get list of all image files
    img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]

    # Randomly select images
    selected_files = random.sample(img_files, num_samples)

    # Set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    # Load models
    print("Loading model trained from scratch...")
    scratch_model = fasterrcnn_resnet50_fpn(num_classes=len(TARGET_CLASSES) + 1)
    scratch_model.load_state_dict(torch.load(scratch_model_path, map_location=device))
    scratch_model.to(device)
    scratch_model.eval()

    print("Loading transfer learning model...")
    transfer_model = fasterrcnn_resnet50_fpn(num_classes=len(TARGET_CLASSES) + 1)
    transfer_model.load_state_dict(torch.load(transfer_model_path, map_location=device))
    transfer_model.to(device)
    transfer_model.eval()

    # Compare model parameters to verify they're different
    scratch_params_sum = sum(p.sum().item() for p in scratch_model.parameters())
    transfer_params_sum = sum(p.sum().item() for p in transfer_model.parameters())
    print(f"Parameter sum check:")
    print(f"  - Scratch model: {scratch_params_sum:.6f}")
    print(f"  - Transfer model: {transfer_params_sum:.6f}")
    print(f"  - Different models: {'Yes' if abs(scratch_params_sum - transfer_params_sum) > 1e-3 else 'No - WARNING!'}")

    # Define transform
    transform = transforms.Compose([transforms.ToTensor()])

    for i, img_file in enumerate(selected_files):
        print(f"\nProcessing image {i + 1}/{num_samples}: {img_file}")
        img_path = os.path.join(img_dir, img_file)
        image_id = os.path.splitext(img_file)[0]
        ann_path = os.path.join(ann_dir, f"{image_id}.xml")

        # Load image
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)

        # Convert image to numpy
        img_np = np.array(img)
        height, width = img_np.shape[:2]

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

        # Get predictions from both models
        with torch.no_grad():
            scratch_preds = scratch_model(img_tensor)[0]
            transfer_preds = transfer_model(img_tensor)[0]

        # Create figure for comparison
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

        # Function to generate heatmap only (no bounding boxes)
        def generate_model_heatmap(predictions, ax, title, color_map=cv2.COLORMAP_JET):
            # Extract predictions
            boxes = predictions['boxes'].cpu().numpy()
            scores = predictions['scores'].cpu().numpy()
            labels = predictions['labels'].cpu().numpy()

            # Filter predictions with confidence > 0.3
            mask = scores > 0.3
            boxes = boxes[mask]
            scores = scores[mask]
            labels = labels[mask]

            # Create a heatmap (confidence map)
            heatmap = np.zeros((height, width))

            # Draw base image first
            ax.imshow(img_np)

            # Add to heatmap
            for box, score, label in zip(boxes, scores, labels):
                # Extract box coordinates
                x1, y1, x2, y2 = map(int, box)

                # Create gaussian blob for confidence
                y, x = np.mgrid[0:height, 0:width]
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                sigma_x = max(5, (x2 - x1) / 3)  # Ensure minimum spread
                sigma_y = max(5, (y2 - y1) / 3)  # Ensure minimum spread

                # Create gaussian
                gaussian = np.exp(-((x - center_x) ** 2 / (2 * sigma_x ** 2) +
                                    (y - center_y) ** 2 / (2 * sigma_y ** 2))) * score

                # Add to heatmap
                heatmap = np.maximum(heatmap, gaussian)

            # Normalize heatmap
            if heatmap.max() > 0:
                heatmap = heatmap / heatmap.max()

            # Apply heatmap as overlay
            heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), color_map)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

            # Create a blend of original image and heatmap
            alpha = 0.7  # Higher alpha for more visible heatmap
            overlay = img_np.copy()
            mask = heatmap > 0.01  # Very low threshold to show more gradients
            overlay[mask] = img_np[mask] * (1 - alpha) + heatmap_colored[mask] * alpha

            # Display overlay without bounding boxes
            ax.imshow(overlay)
            ax.set_title(title)
            ax.axis('off')

            # Return average confidence for title
            return scores.mean() if len(scores) > 0 else 0

        # Generate heatmaps for both models
        scratch_scores = scratch_preds['scores'].cpu().numpy()
        transfer_scores = transfer_preds['scores'].cpu().numpy()

        # Apply the heatmap generation
        scratch_conf = generate_model_heatmap(scratch_preds, axes[1], 'From Scratch')
        transfer_conf = generate_model_heatmap(transfer_preds, axes[2], 'Transfer Learning')

        # Print confidence scores for debugging
        print(f"  Scratch model top 3 confidences: {scratch_scores[:3] if len(scratch_scores) > 0 else 'None'}")
        print(f"  Transfer model top 3 confidences: {transfer_scores[:3] if len(transfer_scores) > 0 else 'None'}")

        # Update titles with average confidence
        axes[1].set_title(f'From Scratch (Avg Conf: {scratch_conf:.2f})')
        axes[2].set_title(f'Transfer Learning (Avg Conf: {transfer_conf:.2f})')

        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'heatmap_comparison_{i + 1}.png'), dpi=200)
        plt.close()

        print(f"  Heatmap saved to {os.path.join(results_dir, f'heatmap_comparison_{i + 1}.png')}")

    print(f"\nAll heatmap visualizations saved to {results_dir}")
    return results_dir


if __name__ == "__main__":
    generate_heatmap_visualizations('dataset_E4888')