import os
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import cv2
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import argparse

# Define target classes
TARGET_CLASSES = ['diningtable', 'sofa']
CLASS_MAPPING = {cls: idx + 1 for idx, cls in enumerate(TARGET_CLASSES)}  # +1 for background
CLASS_MAPPING_INV = {idx: cls for cls, idx in CLASS_MAPPING.items()}


# Custom collate function to avoid pickling issues
def collate_fn(batch):
    return tuple(zip(*batch))


class VOCInstanceSegmentationDataset(Dataset):
    def __init__(self, root, transforms=None, train=True):
        self.root = root
        self.transforms = transforms
        self.train = train

        # Get list of images and annotations
        self.images = []
        self.annotations = []
        self.segmentations = []

        img_dir = os.path.join(root, "images")
        ann_dir = os.path.join(root, "annotations")
        seg_dir = os.path.join(root, "segmentations")  # Folder for segmentation masks

        # Create segmentation directory if it doesn't exist
        os.makedirs(seg_dir, exist_ok=True)

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

                # Check if segmentation exists, otherwise create from bounding boxes
                seg_path = os.path.join(seg_dir, f"{image_id}.png")
                if not os.path.exists(seg_path):
                    self._create_segmentation_mask(img_path, ann_path, seg_path)

                self.segmentations.append(seg_path)

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

    def _create_segmentation_mask(self, img_path, ann_path, seg_path):
        """Create segmentation masks from bounding boxes"""
        # Load image to get dimensions
        img = Image.open(img_path)
        width, height = img.size

        # Create empty mask image
        mask = np.zeros((height, width), dtype=np.uint8)

        # Parse annotation
        tree = ET.parse(ann_path)
        root = tree.getroot()

        class_id = 1  # Start from 1 for instance segmentation

        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name in TARGET_CLASSES:
                # Get bounding box
                bbox = obj.find('bndbox')
                xmin = max(0, int(float(bbox.find('xmin').text)))
                ymin = max(0, int(float(bbox.find('ymin').text)))
                xmax = min(width, int(float(bbox.find('xmax').text)))
                ymax = min(height, int(float(bbox.find('ymax').text)))

                # Skip invalid boxes
                if xmin >= xmax or ymin >= ymax:
                    continue

                # Create simple mask from bounding box (rectangular mask)
                # In a real implementation, you would use actual segmentation masks from dataset
                mask[ymin:ymax, xmin:xmax] = class_id

                # Increment instance ID
                class_id += 1

        # Save mask
        cv2.imwrite(seg_path, mask)

    def __getitem__(self, idx):
        # Load image
        img_path = self.images[idx]
        img = Image.open(img_path).convert("RGB")

        # Get original image dimensions
        orig_width, orig_height = img.size

        # Load mask
        mask_path = self.segmentations[idx]
        mask = Image.open(mask_path)

        # Resize both image and mask to 224x224
        target_size = (224, 224)
        img = img.resize(target_size, Image.BILINEAR)
        mask = mask.resize(target_size, Image.NEAREST)  # Use NEAREST for masks to preserve labels

        mask = np.array(mask)

        # Get unique instance IDs
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[obj_ids > 0]  # Remove background (0)

        # Split the mask into binary masks for each instance
        masks = mask == obj_ids[:, None, None]

        # Load annotation for class labels
        ann_path = self.annotations[idx]
        tree = ET.parse(ann_path)
        root = tree.getroot()

        # Get image dimensions
        width = int(root.find('./size/width').text)
        height = int(root.find('./size/height').text)

        # Get bounding boxes and labels
        boxes = []
        labels = []

        # Calculate scale factors for box coordinates
        scale_x = target_size[0] / orig_width
        scale_y = target_size[1] / orig_height

        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name in TARGET_CLASSES:
                # Get class index
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

                # Scale bounding box coordinates to match the resized image
                xmin = xmin * scale_x
                ymin = ymin * scale_y
                xmax = xmax * scale_x
                ymax = ymax * scale_y

                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(label)

        # Convert to tensors
        if not boxes:  # Handle case with no valid boxes
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)
            masks = torch.zeros((0, target_size[1], target_size[0]), dtype=torch.uint8)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            if len(masks) > 0:
                masks = torch.as_tensor(masks, dtype=torch.uint8)
            else:
                masks = torch.zeros((0, target_size[1], target_size[0]), dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # Assume all instances are not crowd
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
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

                # Flip masks too
                if target["masks"].shape[0] > 0:
                    target["masks"] = target["masks"].flip(-1)

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


def get_instance_segmentation_model(num_classes, pretrained=True):
    """Get Mask R-CNN model with ResNet-50-FPN backbone"""
    # Load pre-trained model
    model = maskrcnn_resnet50_fpn(pretrained=pretrained)

    # Get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256

    # Replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes)

    return model


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


def calculate_metrics(predictions, targets, iou_threshold=0.5):
    """Calculate comprehensive metrics for object detection/segmentation"""

    # Initialize counters
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_correct_class = 0
    total_predictions = 0
    total_gt = 0

    # Per class statistics
    class_stats = {i + 1: {'tp': 0, 'fp': 0, 'fn': 0} for i in range(len(TARGET_CLASSES))}

    # Process each image
    for pred, target in zip(predictions, targets):
        pred_boxes = pred['boxes']
        pred_labels = pred['labels']
        pred_scores = pred['scores']

        gt_boxes = target['boxes']
        gt_labels = target['labels']

        # Track matched ground truth boxes
        matched_gt = [False] * len(gt_boxes)

        # Count total ground truth objects
        total_gt += len(gt_boxes)

        # For each prediction
        for i, (pred_box, pred_label, pred_score) in enumerate(zip(pred_boxes, pred_labels, pred_scores)):
            # Only consider predictions above threshold
            if pred_score < 0.5:
                continue

            total_predictions += 1

            # Find best matching ground truth
            best_iou = 0
            best_gt_idx = -1

            for j, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                # Skip already matched ground truths
                if matched_gt[j]:
                    continue

                # Only compare with same class
                if pred_label == gt_label:
                    iou = calculate_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j

            # Check if we found a match above threshold
            if best_gt_idx >= 0 and best_iou >= iou_threshold:
                total_tp += 1
                if pred_label.item() in class_stats:
                    class_stats[pred_label.item()]['tp'] += 1
                matched_gt[best_gt_idx] = True

                # Correct class prediction
                total_correct_class += 1
            else:
                total_fp += 1
                if pred_label.item() in class_stats:
                    class_stats[pred_label.item()]['fp'] += 1

        # Count false negatives
        for j, matched in enumerate(matched_gt):
            if not matched:
                total_fn += 1
                if gt_labels[j].item() in class_stats:
                    class_stats[gt_labels[j].item()]['fn'] += 1

    # Calculate global metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = total_tp / total_predictions if total_predictions > 0 else 0

    # Calculate per-class metrics
    class_metrics = {}
    for class_id, stats in class_stats.items():
        if class_id - 1 < len(TARGET_CLASSES):
            class_name = TARGET_CLASSES[class_id - 1]

            class_precision = stats['tp'] / (stats['tp'] + stats['fp']) if (stats['tp'] + stats['fp']) > 0 else 0
            class_recall = stats['tp'] / (stats['tp'] + stats['fn']) if (stats['tp'] + stats['fn']) > 0 else 0
            class_f1 = 2 * (class_precision * class_recall) / (class_precision + class_recall) if (
                                                                                                              class_precision + class_recall) > 0 else 0

            class_metrics[class_name] = {
                'precision': class_precision,
                'recall': class_recall,
                'f1': class_f1
            }

    # Calculate mAP
    map_score = calculate_map(predictions, targets, iou_threshold)

    metrics = {
        'map': map_score,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'class_metrics': class_metrics,
        'true_positives': total_tp,
        'false_positives': total_fp,
        'false_negatives': total_fn
    }

    return metrics


def train_model(dataset_path, use_pretrained=True, num_epochs=10, subset_size=1.0,
                early_stopping_patience=3, accumulation_steps=4, batch_size=2):
    # Set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    # Choose model type name for clear identification
    model_type = "transfer" if use_pretrained else "scratch"

    # Define model paths with clear naming
    best_model_path = os.path.join(dataset_path, f'best_{model_type}_segmentation_model.pth')
    final_model_path = os.path.join(dataset_path, f'final_{model_type}_segmentation_model.pth')

    print(f"Model will be saved as:")
    print(f"  - Best model: {best_model_path}")
    print(f"  - Final model: {final_model_path}")

    # Create dataset
    dataset = VOCInstanceSegmentationDataset(dataset_path, transforms=get_transform(train=True), train=True)
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
        batch_size=batch_size,  # Use batch_size parameter
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

    # Create model
    num_classes = len(TARGET_CLASSES) + 1  # +1 for background

    # Get model
    model = get_instance_segmentation_model(num_classes, pretrained=use_pretrained)
    print(f"{'Loading pre-trained' if use_pretrained else 'Training from scratch'} Mask R-CNN model...")

    # Move model to device
    model.to(device)

    # Optimizer with reduced learning rate for stability
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)

    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    # Training loop
    print("Starting training...")
    print(
        f"Using gradient accumulation with {accumulation_steps} steps (effective batch size: {batch_size * accumulation_steps})")
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

        # Zero gradients before the loop
        optimizer.zero_grad()

        for i, (images, targets) in enumerate(tqdm(data_loader_train)):
            batch_count += 1

            # Process images one by one to save memory
            loss_value = 0.0

            try:
                # Process one image at a time
                for img_idx in range(len(images)):
                    single_img = [images[img_idx]]
                    single_target = [targets[img_idx]]

                    # Move to device
                    single_img = [img.to(device) if isinstance(img, torch.Tensor)
                                  else get_transform()(img).to(device) for img in single_img]
                    single_target = [{k: v.to(device) for k, v in t.items()} for t in single_target]

                    # Skip if no boxes
                    if any(len(t['boxes']) == 0 for t in single_target):
                        continue

                    # Forward pass
                    loss_dict = model(single_img, single_target)

                    # Calculate loss and normalize by accumulation steps and batch size
                    losses = sum(loss for loss in loss_dict.values()) / accumulation_steps / len(images)

                    # Backward
                    losses.backward()

                    # Add to loss value
                    loss_value += losses.item() * accumulation_steps * len(images) / len(single_img)

                # Update weights only when accumulation steps are reached
                if (i + 1) % accumulation_steps == 0 or (i + 1) == len(data_loader_train):
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    # Step optimizer
                    optimizer.step()

                    # Zero gradients
                    optimizer.zero_grad()

                    # Clear cache
                    torch.cuda.empty_cache()

                # Update running loss
                running_loss += loss_value

            except Exception as e:
                print(f"Error in training batch: {e}")
                # Clear cache
                torch.cuda.empty_cache()
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

                # Store predictions and targets for metrics calculation
                all_predictions.extend([{k: v.cpu() for k, v in t.items()} for t in outputs])
                all_targets.extend([{k: v.cpu() for k, v in t.items()} for t in targets])

        # Calculate metrics
        val_metrics = calculate_metrics(all_predictions, all_targets)
        val_map = val_metrics['map']
        val_maps.append(val_map)

        # Print metrics
        print(f"Validation Metrics:")
        print(f"  mAP@0.5: {val_map:.4f}")
        print(f"  Precision: {val_metrics['precision']:.4f}")
        print(f"  Recall: {val_metrics['recall']:.4f}")
        print(f"  F1 Score: {val_metrics['f1']:.4f}")
        print(f"  Accuracy: {val_metrics['accuracy']:.4f}")

        # Print per-class metrics
        for class_name, metrics in val_metrics['class_metrics'].items():
            print(
                f"  {class_name}: Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")

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
    plt.savefig(os.path.join(dataset_path, f'segmentation_metrics_{model_type}.png'))
    plt.close()

    return model, train_losses, val_maps


def evaluate_model(model, dataset_path, num_samples=5, confidence_threshold=0.5):
    """Evaluate model on sample images from the dataset"""
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.eval()

    # Create dataset
    dataset = VOCInstanceSegmentationDataset(dataset_path, transforms=get_transform(train=False), train=False)

    # Get sample indices
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

        # Convert to CPU
        prediction = {k: v.cpu() for k, v in prediction.items()}

        # Convert image for visualization
        img_numpy = img.permute(1, 2, 0).cpu().numpy() if isinstance(img, torch.Tensor) else np.array(img)

        # Filter predictions by confidence threshold
        keep_indices = prediction['scores'] >= confidence_threshold
        filtered_boxes = prediction['boxes'][keep_indices]
        filtered_labels = prediction['labels'][keep_indices]
        filtered_scores = prediction['scores'][keep_indices]
        filtered_masks = prediction['masks'][keep_indices]

        # Store for metrics calculation
        all_predictions.append(prediction)
        all_targets.append(target)

        # Calculate IoUs for each prediction with ground truth
        ious = []
        for pred_box, pred_label in zip(filtered_boxes, filtered_labels):
            best_iou = 0
            for gt_box, gt_label in zip(target['boxes'], target['labels']):
                if pred_label == gt_label:
                    iou = calculate_iou(pred_box, gt_box)
                    best_iou = max(best_iou, iou)
            ious.append(best_iou)

        # Calculate precision and recall for this image
        matches = [iou >= 0.5 for iou in ious] if ious else []
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
            'pred_masks': filtered_masks.squeeze(1).numpy(),  # Remove channel dim
            'gt_boxes': target['boxes'].numpy(),
            'gt_labels': target['labels'].numpy(),
            'gt_masks': target['masks'].numpy(),
            'ious': ious,
            'precision': precision,
            'recall': recall
        })

    # Calculate all metrics
    metrics = calculate_metrics(all_predictions, all_targets)

    # Print metrics
    print(f"Evaluation Metrics:")
    print(f"  mAP@0.5: {metrics['map']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1 Score: {metrics['f1']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")

    # Print per-class metrics
    for class_name, class_metrics in metrics['class_metrics'].items():
        print(
            f"  {class_name}: Precision={class_metrics['precision']:.4f}, Recall={class_metrics['recall']:.4f}, F1={class_metrics['f1']:.4f}")

    return results, metrics


def visualize_segmentation_results(results, dataset_path, metrics=None):
    """Visualize instance segmentation results"""
    class_names = TARGET_CLASSES

    # Create results directory if it doesn't exist
    results_dir = dataset_path
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir, exist_ok=True)

    # Create a summary figure
    plt.figure(figsize=(15, 5 * len(results)))

    # Add metrics to title if available
    title = "Instance Segmentation Results"
    if metrics:
        title += f" (mAP: {metrics['map']:.4f}, F1: {metrics['f1']:.4f}, Acc: {metrics['accuracy']:.4f})"

    plt.suptitle(title, fontsize=16)

    for i, result in enumerate(results):
        # Plot ground truth
        plt.subplot(len(results), 2, 2 * i + 1)
        plt.imshow(result['image'])
        plt.title(f'Ground Truth (Image {i + 1})')
        plt.axis('off')

        # Apply masks and boxes for ground truth
        overlay = result['image'].copy()
        # Create a mask label image
        mask_label = np.zeros_like(overlay, dtype=np.uint8)

        for j, (box, label, mask) in enumerate(zip(result['gt_boxes'], result['gt_labels'], result['gt_masks'])):
            # Create colored mask
            mask_color = np.array([0, 255, 0], dtype=np.uint8)  # Green for ground truth
            mask_binary = mask > 0.5
            for c in range(3):  # RGB channels
                mask_label[..., c] = np.where(mask_binary, mask_color[c], mask_label[..., c])

            # Add slight transparency
            overlay = np.where(mask_binary[..., None],
                               overlay * 0.7 + mask_label * 0.3,
                               overlay)

            # Draw bounding box
            x1, y1, x2, y2 = box
            plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                              fill=False, edgecolor='green', linewidth=2))

            cls_name = class_names[label - 1] if label - 1 < len(class_names) else f"Class {label}"
            plt.text(x1, y1, cls_name, bbox=dict(facecolor='green', alpha=0.5))

        plt.imshow(overlay)

        # Plot predictions
        plt.subplot(len(results), 2, 2 * i + 2)
        plt.imshow(result['image'])
        plt.title(f'Predictions (P: {result["precision"]:.2f}, R: {result["recall"]:.2f})')
        plt.axis('off')

        # Apply predicted masks and boxes
        overlay = result['image'].copy()
        mask_label = np.zeros_like(overlay, dtype=np.uint8)

        for j, (box, label, score, mask, iou) in enumerate(zip(
                result['pred_boxes'],
                result['pred_labels'],
                result['pred_scores'],
                result['pred_masks'],
                result['ious'])):

            # Color based on IoU
            if iou >= 0.7:
                mask_color = np.array([0, 255, 0], dtype=np.uint8)  # Green for good IoU
                edge_color = 'lime'
            elif iou >= 0.5:
                mask_color = np.array([255, 255, 0], dtype=np.uint8)  # Yellow for medium IoU
                edge_color = 'yellow'
            else:
                mask_color = np.array([255, 0, 0], dtype=np.uint8)  # Red for poor IoU
                edge_color = 'red'

            # Apply mask with transparency
            mask_binary = mask > 0.5
            for c in range(3):  # RGB channels
                mask_label[..., c] = np.where(mask_binary, mask_color[c], mask_label[..., c])

            overlay = np.where(mask_binary[..., None],
                               overlay * 0.7 + mask_label * 0.3,
                               overlay)

            # Draw bounding box
            x1, y1, x2, y2 = box
            plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                              fill=False, edgecolor=edge_color, linewidth=2))

            cls_name = class_names[label - 1] if label - 1 < len(class_names) else f"Class {label}"
            plt.text(x1, y1, f"{cls_name}: {score:.2f}, IoU: {iou:.2f}",
                     bbox=dict(facecolor=edge_color, alpha=0.5))

        plt.imshow(overlay)

    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust layout for the title
    plt.savefig(os.path.join(results_dir, 'segmentation_results.png'), dpi=200)
    plt.close()

    # Also save individual result images for better detail
    for i, result in enumerate(results):
        plt.figure(figsize=(12, 6))

        # Ground truth visualization
        plt.subplot(1, 2, 1)
        plt.imshow(result['image'])
        plt.title('Ground Truth')
        plt.axis('off')

        # Apply masks with different colors for ground truth
        overlay = result['image'].copy()
        for j, (box, label, mask) in enumerate(zip(result['gt_boxes'], result['gt_labels'], result['gt_masks'])):
            # Create colored mask overlay
            color_mask = np.zeros_like(overlay, dtype=np.uint8)
            mask_binary = mask > 0.5
            color_mask[mask_binary] = [0, 255, 0]  # Green for ground truth

            # Add mask with transparency
            overlay = np.where(mask_binary[..., None],
                               overlay * 0.7 + color_mask * 0.3,
                               overlay)

            # Draw bounding box
            x1, y1, x2, y2 = box
            plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                              fill=False, edgecolor='green', linewidth=2))

            cls_name = class_names[label - 1] if label - 1 < len(class_names) else f"Class {label}"
            plt.text(x1, y1, cls_name, bbox=dict(facecolor='green', alpha=0.5))

        plt.imshow(overlay)

        # Prediction visualization
        plt.subplot(1, 2, 2)
        plt.imshow(result['image'])
        plt.title(f'Predictions (P: {result["precision"]:.2f}, R: {result["recall"]:.2f})')
        plt.axis('off')

        # Apply predicted masks and boxes
        overlay = result['image'].copy()
        for j, (box, label, score, mask, iou) in enumerate(zip(
                result['pred_boxes'],
                result['pred_labels'],
                result['pred_scores'],
                result['pred_masks'],
                result['ious'])):

            # Color based on IoU
            if iou >= 0.7:
                color = [0, 255, 0]  # Green for good IoU
                edge_color = 'lime'
            elif iou >= 0.5:
                color = [255, 255, 0]  # Yellow for medium IoU
                edge_color = 'yellow'
            else:
                color = [255, 0, 0]  # Red for poor IoU
                edge_color = 'red'

            # Apply mask with transparency
            color_mask = np.zeros_like(overlay, dtype=np.uint8)
            mask_binary = mask > 0.5
            color_mask[mask_binary] = color

            overlay = np.where(mask_binary[..., None],
                               overlay * 0.7 + color_mask * 0.3,
                               overlay)

            # Draw bounding box
            x1, y1, x2, y2 = box
            plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                              fill=False, edgecolor=edge_color, linewidth=2))

            cls_name = class_names[label - 1] if label - 1 < len(class_names) else f"Class {label}"
            plt.text(x1, y1, f"{cls_name}: {score:.2f}, IoU: {iou:.2f}",
                     bbox=dict(facecolor=edge_color, alpha=0.5))

        plt.imshow(overlay)

        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'segmentation_result_{i}.png'), dpi=200)
        plt.close()

    print(f"Visualizations saved to {results_dir}")
    return os.path.join(results_dir, 'segmentation_results.png')


def parse_args():
    parser = argparse.ArgumentParser(description='One-Stage Instance Segmentation using Mask R-CNN')
    parser.add_argument('--dataset_path', type=str, default='dataset_E4888', help='Path to the dataset')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training')
    parser.add_argument('--subset_size', type=float, default=1.0,
                        help='Fraction of dataset to use (0.0-1.0). Use 0.5 for 50% of images.')
    parser.add_argument('--skip_scratch', action='store_true', help='Skip training from scratch')
    parser.add_argument('--skip_pretrained', action='store_true', help='Skip training with transfer learning')
    parser.add_argument('--eval_only', action='store_true', help='Only evaluate the model')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold for evaluation')
    parser.add_argument('--accumulation_steps', type=int, default=4,
                        help='Number of gradient accumulation steps (effective batch size multiplier)')
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
            print("\n===== TRAINING SEGMENTATION FROM SCRATCH =====")
            model_scratch, losses_scratch, val_scores_scratch = train_model(
                args.dataset_path, use_pretrained=False, num_epochs=args.epochs,
                subset_size=args.subset_size, accumulation_steps=args.accumulation_steps,
                batch_size=args.batch_size)

        # Train model with pre-trained weights
        if not args.skip_pretrained:
            print("\n===== TRAINING SEGMENTATION WITH TRANSFER LEARNING =====")
            model_pretrained, losses_pretrained, val_scores_pretrained = train_model(
                args.dataset_path, use_pretrained=True, num_epochs=args.epochs,
                subset_size=args.subset_size, accumulation_steps=args.accumulation_steps,
                batch_size=args.batch_size)

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
            comparison_path = os.path.join(args.dataset_path, 'segmentation_comparison.png')
            plt.savefig(comparison_path)
            plt.close()
            print(f"Training comparison saved to {comparison_path}")

    # Evaluation - use best transfer model by default
    if args.eval_only or not args.skip_pretrained:
        print("\n===== EVALUATION =====")

        # Define models to evaluate
        models_to_evaluate = []

        # Add transfer learning model if available
        transfer_model_path = os.path.join(args.dataset_path, 'best_transfer_segmentation_model.pth')
        if os.path.exists(transfer_model_path):
            models_to_evaluate.append(('transfer', transfer_model_path))

        # Add scratch model if available
        scratch_model_path = os.path.join(args.dataset_path, 'best_scratch_segmentation_model.pth')
        if os.path.exists(scratch_model_path):
            models_to_evaluate.append(('scratch', scratch_model_path))

        # Evaluate each available model
        for model_name, model_path in models_to_evaluate:
            print(f"\nEvaluating {model_name} model from: {model_path}")

            # Load model
            model = get_instance_segmentation_model(num_classes=len(TARGET_CLASSES) + 1)
            model.load_state_dict(torch.load(model_path))

            # Run evaluation
            results, metrics = evaluate_model(model, args.dataset_path, confidence_threshold=args.confidence)

            # Save results with model name prefix
            result_dir = os.path.join(args.dataset_path, f'results_segmentation_{model_name}')
            os.makedirs(result_dir, exist_ok=True)

            visualize_segmentation_results(results, result_dir, metrics)
            print(f"Results saved to: {result_dir}")