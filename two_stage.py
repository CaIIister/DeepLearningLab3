import os
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import argparse

# Define target classes - these are the ONLY classes we're working with
TARGET_CLASSES = ['diningtable', 'sofa']
CLASS_MAPPING = {cls: idx + 1 for idx, cls in enumerate(TARGET_CLASSES)}  # Starting from 1 for RCNN
CLASS_MAPPING_INV = {idx: cls for cls, idx in CLASS_MAPPING.items()}  # Inverse mapping


# Custom collate function to avoid pickling issues
def collate_fn(batch):
    return tuple(zip(*batch))


class CustomVOCDataset(Dataset):
    def __init__(self, root, transforms=None, train=True):
        self.root = root
        self.transforms = transforms
        self.train = train

        # Get list of images and annotations
        self.images = []
        self.annotations = []

        img_dir = os.path.join(root, "images")
        ann_dir = os.path.join(root, "annotations")

        # Filter to include only images with target classes
        for ann_file in os.listdir(ann_dir):
            if not ann_file.endswith('.xml'):
                continue

            ann_path = os.path.join(ann_dir, ann_file)
            image_id = os.path.splitext(ann_file)[0]
            img_path = os.path.join(img_dir, f"{image_id}.jpg")

            # Check if image exists and contains target classes
            if os.path.exists(img_path) and self._has_target_classes(ann_path):
                self.images.append(img_path)
                self.annotations.append(ann_path)

        print(f"Loaded {len(self.images)} images with {', '.join(TARGET_CLASSES)}")

    def _has_target_classes(self, ann_path):
        """Check if annotation contains target classes"""
        tree = ET.parse(ann_path)
        root = tree.getroot()

        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name in TARGET_CLASSES:
                return True
        return False

    def __getitem__(self, idx):
        # Load image
        img_path = self.images[idx]
        img = Image.open(img_path).convert("RGB")

        # Load annotation
        ann_path = self.annotations[idx]
        tree = ET.parse(ann_path)
        root = tree.getroot()

        # Get bounding boxes and labels
        boxes = []
        labels = []

        # Get image dimensions
        width = int(root.find('./size/width').text)
        height = int(root.find('./size/height').text)

        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name in TARGET_CLASSES:
                # Get class index (1-indexed for RCNN models)
                label = CLASS_MAPPING[class_name]

                # Get bounding box
                bbox = obj.find('bndbox')
                xmin = max(0, float(bbox.find('xmin').text))
                ymin = max(0, float(bbox.find('ymin').text))
                xmax = min(width, float(bbox.find('xmax').text))
                ymax = min(height, float(bbox.find('ymax').text))

                # Skip invalid boxes
                if xmin >= xmax or ymin >= ymax:
                    continue

                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(label)

        # Convert to tensors
        if not boxes:  # Handle case with no valid boxes
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # Assume all instances are not crowd
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        # Apply data augmentation if training
        if self.train and self.transforms is not None:
            img, target = self.apply_transforms(img, target)
        elif self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def apply_transforms(self, img, target):
        # Convert to tensor first
        img = torchvision.transforms.functional.to_tensor(img)

        # Apply random horizontal flip with 50% probability
        if torch.rand(1) < 0.5:
            img = torchvision.transforms.functional.hflip(img)
            h, w = img.shape[-2:]

            if target["boxes"].shape[0] > 0:
                boxes = target["boxes"].clone()
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
                target["boxes"] = boxes

        return img, target

    def __len__(self):
        return len(self.images)


def get_transform(train=True):
    transforms = []
    transforms.append(torchvision.transforms.ToTensor())

    # Additional transforms for training data
    if train:
        transforms.append(torchvision.transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1))

    return torchvision.transforms.Compose(transforms)


def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two bounding boxes"""
    # Convert to [x1, y1, x2, y2] format
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Calculate area of intersection
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    intersection = w * h

    # Calculate area of union
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection

    # Return IoU
    return intersection / union if union > 0 else 0


def calculate_map(predictions, targets, iou_threshold=0.5):
    """Calculate mean Average Precision at specified IoU threshold"""
    if not predictions or not targets:
        return 0.0

    # Initialize variables for mAP calculation
    aps = []

    # Process each class
    for class_id in range(1, len(TARGET_CLASSES) + 1):
        # Extract all predictions and ground truths for this class
        class_preds = []
        class_targets = []

        for i, (pred, target) in enumerate(zip(predictions, targets)):
            # Get predictions for this class
            pred_indices = (pred['labels'] == class_id).nonzero(as_tuple=True)[0]
            pred_boxes = pred['boxes'][pred_indices]
            pred_scores = pred['scores'][pred_indices]

            # Get ground truths for this class
            target_indices = (target['labels'] == class_id).nonzero(as_tuple=True)[0]
            target_boxes = target['boxes'][target_indices]

            # Store predictions with scores and ground truths
            # Use image index i instead of relying on 'image_id'
            for box, score in zip(pred_boxes, pred_scores):
                class_preds.append((box, score, i))

            for box in target_boxes:
                class_targets.append((box, i))

        # Skip if no predictions or targets for this class
        if not class_preds or not class_targets:
            continue

        # Sort predictions by confidence score (highest first)
        class_preds.sort(key=lambda x: x[1], reverse=True)

        # Initialize variables for precision-recall calculation
        tp = np.zeros(len(class_preds))
        fp = np.zeros(len(class_preds))
        gt_covered = set()

        # For each prediction, check if it's a true positive
        for i, (pred_box, _, img_idx) in enumerate(class_preds):
            # Find the best matching ground truth
            best_iou = 0
            best_gt_idx = -1

            for j, (gt_box, gt_img_idx) in enumerate(class_targets):
                # Only compare boxes from the same image
                if img_idx != gt_img_idx:
                    continue

                # Skip already covered ground truths
                if (gt_img_idx, j) in gt_covered:
                    continue

                # Calculate IoU
                iou = calculate_iou(pred_box, gt_box)

                # Update best match
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_gt_idx = j

            # Check if we found a match
            if best_gt_idx >= 0:
                tp[i] = 1
                gt_covered.add((class_targets[best_gt_idx][1], best_gt_idx))
            else:
                fp[i] = 1

        # Calculate cumulative sums
        cumsum_tp = np.cumsum(tp)
        cumsum_fp = np.cumsum(fp)

        # Calculate precision and recall
        precision = cumsum_tp / (cumsum_tp + cumsum_fp)
        recall = cumsum_tp / len(class_targets)

        # Calculate average precision
        ap = 0
        for t in np.arange(0, 1.1, 0.1):
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                p = np.max(precision[recall >= t])
            ap += p / 11

        aps.append(ap)

    # Calculate mAP
    return np.mean(aps) if aps else 0.0


def train_model(dataset_path, use_pretrained=True, num_epochs=10, subset_size=1.0, early_stopping_patience=3):
    # Set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    # Choose model type name for clear identification
    model_type = "transfer" if use_pretrained else "scratch"

    # Define model paths with clear naming
    best_model_path = os.path.join(dataset_path, f'best_{model_type}_model.pth')
    final_model_path = os.path.join(dataset_path, f'final_{model_type}_model.pth')

    print(f"Model will be saved as:")
    print(f"  - Best model: {best_model_path}")
    print(f"  - Final model: {final_model_path}")

    # Create dataset
    dataset = CustomVOCDataset(dataset_path, transforms=get_transform(train=True), train=True)
    full_size = len(dataset)
    print(f"Full dataset size: {full_size} images")

    # Apply subset size if less than 1.0
    if subset_size < 1.0:
        subset_size = max(0.1, min(1.0, subset_size))  # Ensure it's between 0.1 and 1.0
        reduced_size = int(full_size * subset_size)
        indices = torch.randperm(full_size)[:reduced_size].tolist()
        dataset = torch.utils.data.Subset(dataset, indices)
        print(f"Using {subset_size:.0%} of dataset: {len(dataset)} images")

    # Split dataset into train and validation (80/20 split)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    indices = torch.randperm(len(dataset)).tolist()
    dataset_train = torch.utils.data.Subset(dataset, indices[:train_size])
    dataset_val = torch.utils.data.Subset(dataset, indices[train_size:])

    print(f"Training set: {len(dataset_train)} images")
    print(f"Validation set: {len(dataset_val)} images")

    # Dataloaders
    data_loader_train = DataLoader(
        dataset_train,
        batch_size=2,
        shuffle=True,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        collate_fn=collate_fn
    )

    data_loader_val = DataLoader(
        dataset_val,
        batch_size=1,
        shuffle=False,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        collate_fn=collate_fn
    )

    # Create model - two-stage Faster R-CNN with ResNet50 backbone
    num_classes = len(TARGET_CLASSES) + 1  # +1 for background class

    if use_pretrained:
        print("Loading pre-trained model...")
        # Load pre-trained model with default weights
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn(weights=weights)

        # Replace classification head with a new one for our classes
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features,
                                                                                                   num_classes)
    else:
        print("Training model from scratch...")
        # Create model without pre-trained weights
        model = fasterrcnn_resnet50_fpn(weights=None, num_classes=num_classes)

    # Move model to device
    model.to(device)

    # Optimizer with reduced learning rate for stability
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    # Training loop
    print("Starting training...")
    start_time = time.time()

    # Track metrics
    train_losses = []
    val_maps = []
    best_map = 0.0
    patience_counter = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Training phase
        model.train()
        running_loss = 0.0
        batch_count = 0

        for images, targets in tqdm(data_loader_train):
            batch_count += 1

            # Move data to device
            images = list(image.to(device) if isinstance(image, torch.Tensor)
                          else get_transform()(image).to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Skip batches with no boxes
            if any(len(t['boxes']) == 0 for t in targets):
                print("Skipping batch with empty targets")
                continue

            # Reset gradients
            optimizer.zero_grad()

            try:
                # Forward pass
                loss_dict = model(images, targets)

                # Calculate standard loss (sum of all loss components)
                losses = sum(loss for loss in loss_dict.values())
                loss_value = losses.item()

                # Check for valid loss
                if not torch.isfinite(losses):
                    print(f"Warning: Non-finite loss {loss_value}")
                    continue

                # Backward pass and optimize
                losses.backward()

                # Add gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                # Update running loss
                running_loss += loss_value

            except Exception as e:
                print(f"Error in training batch: {e}")
                continue

        # Calculate average training loss
        epoch_loss = running_loss / max(1, batch_count)
        train_losses.append(epoch_loss)
        print(f"Training Loss: {epoch_loss:.4f}")

        # Validation phase
        model.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for images, targets in tqdm(data_loader_val):
                images = list(image.to(device) if isinstance(image, torch.Tensor)
                              else get_transform(train=False)(image).to(device) for image in images)

                # Run model
                outputs = model(images)

                # Store predictions and targets for mAP calculation
                all_predictions.extend([{k: v.cpu() for k, v in t.items()} for t in outputs])
                all_targets.extend([{k: v.cpu() for k, v in t.items()} for t in targets])

        # Calculate mAP
        val_map = calculate_map(all_predictions, all_targets)
        val_maps.append(val_map)
        print(f"Validation mAP@0.5: {val_map:.4f}")

        # Update learning rate scheduler
        lr_scheduler.step(1.0 - val_map)  # Use inverted mAP as "loss" to minimize

        # Check for improvement
        if val_map > best_map:
            best_map = val_map
            patience_counter = 0
            print(f"New best mAP: {best_map:.4f} - Saving {model_type} model")
            # Save best model
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs. Best mAP: {best_map:.4f}")

        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    # Training complete
    total_time = time.time() - start_time
    print(f"Training complete in {total_time / 60:.2f} minutes")

    # Load best model
    if os.path.exists(best_model_path):
        print(f"Loading best model with mAP: {best_map:.4f}")
        model.load_state_dict(torch.load(best_model_path))

    # Save final model
    print(f"Saving final {model_type} model")
    torch.save(model.state_dict(), final_model_path)

    # Plot training loss and validation mAP
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title(f'Training Loss ({model_type})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(val_maps)
    plt.title(f'Validation mAP@0.5 ({model_type})')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')

    plt.tight_layout()
    plt.savefig(os.path.join(dataset_path, f'training_metrics_{model_type}.png'))
    plt.close()

    return model, train_losses, val_maps


def evaluate_model(model, dataset_path, num_samples=5, confidence_threshold=0.5):
    """Evaluate model on sample images from the dataset with improved metrics"""
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.eval()

    # Create dataset
    dataset = CustomVOCDataset(dataset_path, transforms=get_transform(train=False), train=False)

    # Get sample indices, either random or deterministic depending on implementation needs
    indices = torch.randperm(len(dataset))[:num_samples].tolist()

    results = []
    all_predictions = []
    all_targets = []

    for idx in indices:
        # Get image and target
        img, target = dataset[idx]

        # Make prediction
        with torch.no_grad():
            img_tensor = img if isinstance(img, torch.Tensor) else get_transform(train=False)(img)
            prediction = model([img_tensor.to(device)])[0]

        # Convert tensors to CPU for processing
        prediction = {k: v.cpu() for k, v in prediction.items()}

        # Convert image for visualization
        img_numpy = img.permute(1, 2, 0).cpu().numpy() if isinstance(img, torch.Tensor) else np.array(img)

        # Filter predictions by confidence threshold
        keep_indices = prediction['scores'] >= confidence_threshold
        filtered_boxes = prediction['boxes'][keep_indices]
        filtered_labels = prediction['labels'][keep_indices]
        filtered_scores = prediction['scores'][keep_indices]

        # Store for metrics calculation
        all_predictions.append(prediction)
        all_targets.append(target)

        # Calculate IoUs for each prediction with ground truth
        ious = []
        matches = []

        for pred_box, pred_label in zip(filtered_boxes, filtered_labels):
            best_iou = 0
            for gt_box, gt_label in zip(target['boxes'], target['labels']):
                if pred_label == gt_label:
                    iou = calculate_iou(pred_box, gt_box)
                    best_iou = max(best_iou, iou)
            ious.append(best_iou)
            matches.append(best_iou >= 0.5)  # True if IoU >= 0.5

        # Calculate precision and recall for this image
        tp = sum(matches)
        fp = len(matches) - tp
        fn = len(target['boxes']) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        # Store result
        results.append({
            'image': img_numpy,
            'pred_boxes': filtered_boxes.numpy(),
            'pred_labels': filtered_labels.numpy(),
            'pred_scores': filtered_scores.numpy(),
            'gt_boxes': target['boxes'].numpy(),
            'gt_labels': target['labels'].numpy(),
            'ious': ious,
            'precision': precision,
            'recall': recall
        })

    # Calculate overall metrics
    map_score = calculate_map(all_predictions, all_targets)

    # Print metrics
    print(f"Evaluation Metrics:")
    print(f"  mAP@0.5: {map_score:.4f}")

    avg_precision = np.mean([r['precision'] for r in results])
    avg_recall = np.mean([r['recall'] for r in results])

    print(f"  Average Precision: {avg_precision:.4f}")
    print(f"  Average Recall: {avg_recall:.4f}")

    metrics = {
        'map': map_score,
        'precision': avg_precision,
        'recall': avg_recall
    }

    return results, metrics


def visualize_results(results, dataset_path, metrics=None):
    """Visualize detection results with improved formatting"""
    class_names = TARGET_CLASSES

    # Create results directory if it doesn't exist
    results_dir = dataset_path
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir, exist_ok=True)

    # Create a summary figure with all results
    plt.figure(figsize=(15, 5 * len(results)))

    # Add metrics to title if available
    title = "Object Detection Results"
    if metrics:
        title += f" (mAP@0.5: {metrics['map']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f})"

    plt.suptitle(title, fontsize=16)

    for i, result in enumerate(results):
        # Plot ground truth
        plt.subplot(len(results), 2, 2 * i + 1)
        plt.imshow(result['image'])
        plt.title(f'Ground Truth (Image {i + 1})')
        plt.axis('off')

        for box, label in zip(result['gt_boxes'], result['gt_labels']):
            x1, y1, x2, y2 = box
            plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                              fill=False, edgecolor='green', linewidth=2))
            cls_name = class_names[label - 1] if label - 1 < len(class_names) else f"Class {label}"
            plt.text(x1, y1, cls_name,
                     bbox=dict(facecolor='green', alpha=0.5))

        # Plot predictions
        plt.subplot(len(results), 2, 2 * i + 2)
        plt.imshow(result['image'])
        plt.title(f'Predictions (P: {result["precision"]:.2f}, R: {result["recall"]:.2f})')
        plt.axis('off')

        for j, (box, label, score) in enumerate(zip(result['pred_boxes'],
                                                    result['pred_labels'],
                                                    result['pred_scores'])):
            x1, y1, x2, y2 = box

            # Color based on IoU if available
            color = 'red'
            if 'ious' in result and j < len(result['ious']):
                iou = result['ious'][j]
                # Green for good matches, yellow for okay, red for poor
                if iou >= 0.7:
                    color = 'lime'
                elif iou >= 0.5:
                    color = 'yellow'

            plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                              fill=False, edgecolor=color, linewidth=2))

            cls_name = class_names[label - 1] if label - 1 < len(class_names) else f"Class {label}"
            iou_text = f", IoU: {result['ious'][j]:.2f}" if 'ious' in result and j < len(result['ious']) else ""

            plt.text(x1, y1, f"{cls_name}: {score:.2f}{iou_text}",
                     bbox=dict(facecolor=color, alpha=0.5))

    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust layout for the title
    plt.savefig(os.path.join(results_dir, 'detection_results.png'), dpi=200)
    plt.close()

    # Also save individual result images for better detail
    for i, result in enumerate(results):
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.imshow(result['image'])
        plt.title('Ground Truth')
        plt.axis('off')

        for box, label in zip(result['gt_boxes'], result['gt_labels']):
            x1, y1, x2, y2 = box
            plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                              fill=False, edgecolor='green', linewidth=2))
            cls_name = class_names[label - 1] if label - 1 < len(class_names) else f"Class {label}"
            plt.text(x1, y1, cls_name,
                     bbox=dict(facecolor='green', alpha=0.5))

        plt.subplot(1, 2, 2)
        plt.imshow(result['image'])
        plt.title(f'Predictions (P: {result["precision"]:.2f}, R: {result["recall"]:.2f})')
        plt.axis('off')

        for j, (box, label, score) in enumerate(zip(result['pred_boxes'],
                                                    result['pred_labels'],
                                                    result['pred_scores'])):
            x1, y1, x2, y2 = box

            # Color based on IoU
            color = 'red'
            if 'ious' in result and j < len(result['ious']):
                iou = result['ious'][j]
                if iou >= 0.7:
                    color = 'lime'
                elif iou >= 0.5:
                    color = 'yellow'

            plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                              fill=False, edgecolor=color, linewidth=2))

            cls_name = class_names[label - 1] if label - 1 < len(class_names) else f"Class {label}"
            plt.text(x1, y1, f"{cls_name}: {score:.2f}, IoU: {result['ious'][j]:.2f}",
                     bbox=dict(facecolor=color, alpha=0.5))

        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'result_{i}.png'), dpi=200)
        plt.close()

    print(f"Visualizations saved to {results_dir}")
    return os.path.join(results_dir, 'detection_results.png')


def parse_args():
    parser = argparse.ArgumentParser(description='Two-Stage Object Detection using Faster R-CNN')
    parser.add_argument('--dataset_path', type=str, default='dataset_E4888', help='Path to the dataset')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training')
    parser.add_argument('--subset_size', type=float, default=1.0,
                        help='Fraction of dataset to use (0.0-1.0). Use 0.5 for 50% of images.')
    parser.add_argument('--skip_scratch', action='store_true', help='Skip training from scratch')
    parser.add_argument('--skip_pretrained', action='store_true', help='Skip training with transfer learning')
    parser.add_argument('--eval_only', action='store_true', help='Only evaluate the model')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold for evaluation')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Create output directories if they don't exist
    os.makedirs(args.dataset_path, exist_ok=True)
    os.makedirs(os.path.join(args.dataset_path, 'models'), exist_ok=True)
    os.makedirs(os.path.join(args.dataset_path, 'results'), exist_ok=True)

    # By default, train both model types unless specifically skipped
    if not args.eval_only:
        # Train model from scratch
        if not args.skip_scratch:
            print("\n===== TRAINING FROM SCRATCH =====")
            model_scratch, losses_scratch, val_scores_scratch = train_model(
                args.dataset_path, use_pretrained=False, num_epochs=args.epochs, subset_size=args.subset_size)

        # Train model with pre-trained weights
        if not args.skip_pretrained:
            print("\n===== TRAINING WITH TRANSFER LEARNING =====")
            model_pretrained, losses_pretrained, val_scores_pretrained = train_model(
                args.dataset_path, use_pretrained=True, num_epochs=args.epochs, subset_size=args.subset_size)

        # Compare both models if both were trained
        if not args.skip_scratch and not args.skip_pretrained:
            print("\n===== COMPARING TRAINING APPROACHES =====")
            plt.figure(figsize=(12, 6))

            plt.subplot(1, 2, 1)
            plt.plot(losses_scratch, label='From Scratch')
            plt.plot(losses_pretrained, label='Transfer Learning')
            plt.title('Training Loss Comparison')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(val_scores_scratch, label='From Scratch')
            plt.plot(val_scores_pretrained, label='Transfer Learning')
            plt.title('Validation mAP Comparison')
            plt.xlabel('Epoch')
            plt.ylabel('mAP@0.5')
            plt.legend()

            plt.tight_layout()
            comparison_path = os.path.join(args.dataset_path, 'training_comparison.png')
            plt.savefig(comparison_path)
            plt.close()
            print(f"Training comparison saved to {comparison_path}")

    # Evaluation - use best transfer model by default
    if args.eval_only or not args.skip_pretrained:
        print("\n===== EVALUATION =====")

        # Define models to evaluate
        models_to_evaluate = []

        # Add transfer learning model if available
        transfer_model_path = os.path.join(args.dataset_path, 'best_transfer_model.pth')
        if os.path.exists(transfer_model_path):
            models_to_evaluate.append(('transfer', transfer_model_path))

        # Add scratch model if available
        scratch_model_path = os.path.join(args.dataset_path, 'best_scratch_model.pth')
        if os.path.exists(scratch_model_path):
            models_to_evaluate.append(('scratch', scratch_model_path))

        # Evaluate each available model
        for model_name, model_path in models_to_evaluate:
            print(f"\nEvaluating {model_name} model from: {model_path}")

            # Load model
            model = fasterrcnn_resnet50_fpn(num_classes=len(TARGET_CLASSES) + 1)
            model.load_state_dict(torch.load(model_path))

            # Run evaluation
            results, metrics = evaluate_model(model, args.dataset_path, confidence_threshold=args.confidence)

            # Save results with model name prefix
            result_dir = os.path.join(args.dataset_path, f'results_{model_name}')
            os.makedirs(result_dir, exist_ok=True)

            visualize_results(results, result_dir, metrics)
            print(f"Results saved to: {result_dir}")