import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import xml.etree.ElementTree as ET
import cv2
import argparse

# Import the model from one_stage.py
from one_stage import get_instance_segmentation_model, TARGET_CLASSES


def generate_bbox_visualizations(dataset_path, num_samples=3, confidence_threshold=0.5):
    """
    Generate bounding box visualizations comparing models trained from scratch vs transfer learning.
    Shows masks and bounding boxes for instance segmentation.
    """
    # Paths to the trained models
    scratch_model_path = os.path.join(dataset_path, 'best_scratch_segmentation_model.pth')
    transfer_model_path = os.path.join(dataset_path, 'best_transfer_segmentation_model.pth')

    # Verify model paths exist
    if not os.path.exists(scratch_model_path):
        raise FileNotFoundError(f"Scratch model not found at {scratch_model_path}")
    if not os.path.exists(transfer_model_path):
        raise FileNotFoundError(f"Transfer model not found at {transfer_model_path}")

    print(f"Loading models from:")
    print(f"  - Scratch: {scratch_model_path}")
    print(f"  - Transfer: {transfer_model_path}")

    # Create results directory
    results_dir = os.path.join(dataset_path, 'results', 'bbox_comparison')
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
    print("Loading models...")
    # Loading model trained from scratch
    scratch_model = get_instance_segmentation_model(num_classes=len(TARGET_CLASSES) + 1)
    scratch_model.load_state_dict(torch.load(scratch_model_path, map_location=device))
    scratch_model.to(device)
    scratch_model.eval()

    # Loading transfer learning model
    transfer_model = get_instance_segmentation_model(num_classes=len(TARGET_CLASSES) + 1)
    transfer_model.load_state_dict(torch.load(transfer_model_path, map_location=device))
    transfer_model.to(device)
    transfer_model.eval()

    # Verify models are different
    scratch_params = sum(p.sum().item() for p in scratch_model.parameters())
    transfer_params = sum(p.sum().item() for p in transfer_model.parameters())
    print(f"Parameter sum check:")
    print(f"  - Scratch model: {scratch_params:.4f}")
    print(f"  - Transfer model: {transfer_params:.4f}")
    print(f"  - Different: {abs(scratch_params - transfer_params) > 1e-3}")

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
                gt_labels.append(TARGET_CLASSES.index(class_name))

        # Convert image to numpy for visualization
        img_np = np.array(img)

        # Get predictions from both models
        with torch.no_grad():
            scratch_preds = scratch_model(img_tensor)[0]
            transfer_preds = transfer_model(img_tensor)[0]

        # Filter predictions by confidence threshold
        scratch_scores = scratch_preds['scores'].cpu().numpy()
        scratch_keep = scratch_scores > confidence_threshold
        scratch_boxes = scratch_preds['boxes'].cpu().numpy()[scratch_keep]
        scratch_labels = scratch_preds['labels'].cpu().numpy()[
                             scratch_keep] - 1  # -1 to map back to TARGET_CLASSES index
        scratch_scores = scratch_scores[scratch_keep]
        scratch_masks = scratch_preds['masks'].cpu().numpy()[scratch_keep, 0]  # First channel contains the mask

        transfer_scores = transfer_preds['scores'].cpu().numpy()
        transfer_keep = transfer_scores > confidence_threshold
        transfer_boxes = transfer_preds['boxes'].cpu().numpy()[transfer_keep]
        transfer_labels = transfer_preds['labels'].cpu().numpy()[
                              transfer_keep] - 1  # -1 to map back to TARGET_CLASSES index
        transfer_scores = transfer_scores[transfer_keep]
        transfer_masks = transfer_preds['masks'].cpu().numpy()[transfer_keep, 0]  # First channel contains the mask

        # Calculate average confidence scores
        scratch_avg_conf = scratch_scores.mean() if len(scratch_scores) > 0 else 0
        transfer_avg_conf = transfer_scores.mean() if len(transfer_scores) > 0 else 0

        # Create figure with three subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Ground Truth
        axes[0].imshow(img_np)
        axes[0].set_title('Ground Truth')
        axes[0].axis('off')

        for box, label_idx in zip(gt_boxes, gt_labels):
            xmin, ymin, xmax, ymax = box
            rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                 fill=False, edgecolor='green', linewidth=2)
            axes[0].add_patch(rect)
            axes[0].text(xmin, ymin, TARGET_CLASSES[label_idx],
                         bbox=dict(facecolor='green', alpha=0.5), fontsize=10)

        # Model from Scratch
        axes[1].imshow(img_np)
        axes[1].set_title(f'From Scratch (Avg Conf: {scratch_avg_conf:.2f})')
        axes[1].axis('off')

        # Create a blend of the image and the mask predictions for scratch model
        scratch_overlay = img_np.copy()
        for box, label_idx, score, mask in zip(scratch_boxes, scratch_labels, scratch_scores, scratch_masks):
            # Apply mask with transparency
            mask_binary = mask > 0.5
            mask_color = np.zeros_like(scratch_overlay, dtype=np.uint8)
            mask_color[mask_binary] = [255, 0, 0]  # Red for scratch model

            # Blend image and mask
            scratch_overlay = np.where(
                mask_binary[..., None],
                scratch_overlay * 0.7 + mask_color * 0.3,
                scratch_overlay
            )

            # Draw bounding box
            xmin, ymin, xmax, ymax = box
            rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                 fill=False, edgecolor='red', linewidth=2)
            axes[1].add_patch(rect)

            # Add label and confidence
            class_name = TARGET_CLASSES[label_idx] if 0 <= label_idx < len(TARGET_CLASSES) else f"Class {label_idx + 1}"
            axes[1].text(xmin, ymin, f"{class_name}: {score:.2f}",
                         bbox=dict(facecolor='red', alpha=0.5), fontsize=10)

        axes[1].imshow(scratch_overlay)

        # Transfer Learning
        axes[2].imshow(img_np)
        axes[2].set_title(f'Transfer Learning (Avg Conf: {transfer_avg_conf:.2f})')
        axes[2].axis('off')

        # Create a blend of the image and the mask predictions for transfer model
        transfer_overlay = img_np.copy()
        for box, label_idx, score, mask in zip(transfer_boxes, transfer_labels, transfer_scores, transfer_masks):
            # Apply mask with transparency
            mask_binary = mask > 0.5
            mask_color = np.zeros_like(transfer_overlay, dtype=np.uint8)
            mask_color[mask_binary] = [0, 0, 255]  # Blue for transfer model

            # Blend image and mask
            transfer_overlay = np.where(
                mask_binary[..., None],
                transfer_overlay * 0.7 + mask_color * 0.3,
                transfer_overlay
            )

            # Draw bounding box
            xmin, ymin, xmax, ymax = box
            rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                 fill=False, edgecolor='blue', linewidth=2)
            axes[2].add_patch(rect)

            # Add label and confidence
            class_name = TARGET_CLASSES[label_idx] if 0 <= label_idx < len(TARGET_CLASSES) else f"Class {label_idx + 1}"
            axes[2].text(xmin, ymin, f"{class_name}: {score:.2f}",
                         bbox=dict(facecolor='blue', alpha=0.5), fontsize=10)

        axes[2].imshow(transfer_overlay)

        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'bbox_comparison_{i + 1}.png'), dpi=200)
        plt.close()

        print(f"  Image {i + 1} saved to {os.path.join(results_dir, f'bbox_comparison_{i + 1}.png')}")

    print(f"All visualizations saved to {results_dir}")
    return results_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate bounding box visualizations')
    parser.add_argument('--dataset_path', type=str, default='dataset_E4888', help='Path to the dataset')
    parser.add_argument('--num_samples', type=int, default=3, help='Number of sample images to visualize')
    args = parser.parse_args()

    generate_bbox_visualizations(args.dataset_path, args.num_samples)