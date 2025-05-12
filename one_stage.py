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


# Custom collate function
def collate_fn(batch):
    return tuple(zip(*batch))


class VOCInstanceSegmentationDataset(Dataset):
    def __init__(self, root, transforms=None, train=True):
        self.root = root
        self.transforms = transforms
        self.train = train

        # Get paths to images and annotations
        self.images = []
        self.annotations = []
        self.segmentations = []

        img_dir = os.path.join(root, "images")
        ann_dir = os.path.join(root, "annotations")
        seg_dir = os.path.join(root, "segmentations")  # Folder for segmentation masks

        # Create segmentation directory if it doesn't exist
        os.makedirs(seg_dir, exist_ok=True)

        # Find images with target classes
        for ann_file in os.listdir(ann_dir):
            if not ann_file.endswith('.xml'):
                continue

            ann_path = os.path.join(ann_dir, ann_file)
            image_id = os.path.splitext(ann_file)[0]
            img_path = os.path.join(img_dir, f"{image_id}.jpg")

            # Only include images that exist and have target classes
            if os.path.exists(img_path) and self._has_target_classes(ann_path):
                self.images.append(img_path)
                self.annotations.append(ann_path)

                # Check if segmentation exists, create if not
                seg_path = os.path.join(seg_dir, f"{image_id}.png")
                if not os.path.exists(seg_path):
                    self._create_segmentation_mask(img_path, ann_path, seg_path)

                self.segmentations.append(seg_path)

        print(f"Loaded {len(self.images)} images with {', '.join(TARGET_CLASSES)}")

        # Count class instances for balancing
        self.class_counts = self._count_class_instances()

    def _count_class_instances(self):
        """Count instances of each class"""
        class_counts = {cls: 0 for cls in TARGET_CLASSES}

        for ann_path in self.annotations:
            tree = ET.parse(ann_path)
            root = tree.getroot()

            for obj in root.findall('object'):
                class_name = obj.find('name').text
                if class_name in TARGET_CLASSES:
                    class_counts[class_name] += 1

        return class_counts

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
        img = cv2.imread(img_path)
        if img is None:
            pil_img = Image.open(img_path)
            width, height = pil_img.size
            mask = np.zeros((height, width), dtype=np.uint8)
            self._create_simple_mask(ann_path, mask, width, height)
        else:
            # Create empty mask image
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            height, width = mask.shape
            self._create_simple_mask(ann_path, mask, width, height)

        # Save mask
        cv2.imwrite(seg_path, mask)

    def _create_simple_mask(self, ann_path, mask, width, height):
        """Create simple rectangular masks from bounding boxes"""
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

                # Create mask from bounding box
                mask[ymin:ymax, xmin:xmax] = class_id
                class_id += 1

    def __getitem__(self, idx):
        # Load image
        img_path = self.images[idx]
        img = Image.open(img_path).convert("RGB")

        # Get original image dimensions
        orig_width, orig_height = img.size

        # Load mask
        mask_path = self.segmentations[idx]
        mask = Image.open(mask_path)

        # Resize to standard size
        target_size = (512, 512)
        img = img.resize(target_size, Image.BILINEAR)
        mask = mask.resize(target_size, Image.NEAREST)  # Use NEAREST for masks

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
                xmax = min(float(root.find('./size/width').text), float(bbox.find('xmax').text))
                ymax = min(float(root.find('./size/height').text), float(bbox.find('ymax').text))

                # Skip invalid boxes
                if xmin >= xmax or ymin >= ymax:
                    continue

                # Scale bounding box coordinates
                xmin = xmin * scale_x
                ymin = ymin * scale_y
                xmax = xmax * scale_x
                ymax = ymax * scale_y

                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(label)

        # Convert to tensors
        if not boxes:  # Handle empty case
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

        # Create target dictionary
        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([idx]),
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64)
        }

        # Apply transforms
        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.images)


def get_transform(train=True):
    """Get basic transforms for training/testing"""
    transforms = []
    transforms.append(torchvision.transforms.ToTensor())

    # Add color jitter for training
    if train:
        transforms.append(torchvision.transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1))

    return torchvision.transforms.Compose(transforms)


def get_instance_segmentation_model(num_classes, pretrained=True):
    """Get Mask R-CNN model"""
    # Load model
    model = maskrcnn_resnet50_fpn(pretrained=pretrained)

    # Replace box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Replace mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model


def calculate_iou(box1, box2):
    """Calculate Intersection over Union"""
    # Find intersection coordinates
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
    """Calculate mean Average Precision"""
    if not predictions or not targets:
        return 0.0

    # Calculate AP for each class
    aps = []

    # Process each class
    for class_id in range(1, len(TARGET_CLASSES) + 1):
        # Get all predictions and ground truths for this class
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

            # Store with image index
            for box, score in zip(pred_boxes, pred_scores):
                class_preds.append((box, score, i))

            for box in target_boxes:
                class_targets.append((box, i))

        # Skip if no predictions or targets
        if not class_preds or not class_targets:
            continue

        # Sort predictions by confidence (highest first)
        class_preds.sort(key=lambda x: x[1], reverse=True)

        # Calculate precision-recall
        tp = np.zeros(len(class_preds))
        fp = np.zeros(len(class_preds))
        gt_covered = set()

        # Check each prediction
        for i, (pred_box, _, img_idx) in enumerate(class_preds):
            # Find best matching ground truth
            best_iou = 0
            best_gt_idx = -1

            for j, (gt_box, gt_img_idx) in enumerate(class_targets):
                # Only compare boxes from same image
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

        # Calculate precision and recall
        cumsum_tp = np.cumsum(tp)
        cumsum_fp = np.cumsum(fp)

        precision = cumsum_tp / (cumsum_tp + cumsum_fp)
        recall = cumsum_tp / len(class_targets)

        # Calculate AP using 11-point interpolation
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


def train_model(dataset_path, use_pretrained=True, num_epochs=10):
    """Train Mask R-CNN model"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Model type for file naming
    model_type = "transfer" if use_pretrained else "scratch"
    best_model_path = os.path.join(dataset_path, f'best_{model_type}_segmentation_model.pth')

    # Create dataset
    dataset = VOCInstanceSegmentationDataset(dataset_path, transforms=get_transform(train=True), train=True)

    # Split into train and validation (80/20)
    train_size = int(0.8 * len(dataset))
    indices = torch.randperm(len(dataset)).tolist()
    dataset_train = torch.utils.data.Subset(dataset, indices[:train_size])
    dataset_val = torch.utils.data.Subset(dataset, indices[train_size:])

    print(f"Training set: {len(dataset_train)} images")
    print(f"Validation set: {len(dataset_val)} images")

    # Create data loaders
    data_loader_train = DataLoader(
        dataset_train, batch_size=2, shuffle=True,
        num_workers=0, collate_fn=collate_fn
    )

    data_loader_val = DataLoader(
        dataset_val, batch_size=1, shuffle=False,
        num_workers=0, collate_fn=collate_fn
    )

    # Create model
    num_classes = len(TARGET_CLASSES) + 1  # +1 for background
    model = get_instance_segmentation_model(num_classes, pretrained=use_pretrained)
    model.to(device)

    # Create optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Create scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Training loop
    start_time = time.time()
    best_map = 0.0
    train_losses = []
    val_maps = []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Training phase
        model.train()
        running_loss = 0.0

        for images, targets in tqdm(data_loader_train):
            # Move to device
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # Backward pass
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            # Update running loss
            running_loss += losses.item()

        # Update learning rate
        lr_scheduler.step()

        # Calculate average loss
        epoch_loss = running_loss / len(data_loader_train)
        train_losses.append(epoch_loss)
        print(f"Training Loss: {epoch_loss:.4f}")

        # Validation phase
        model.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for images, targets in tqdm(data_loader_val):
                # Move images to device
                images = [img.to(device) for img in images]

                # Forward pass
                outputs = model(images)

                # Store outputs and targets
                all_predictions.extend([{k: v.cpu() for k, v in t.items()} for t in outputs])
                all_targets.extend(targets)

        # Calculate mAP
        val_map = calculate_map(all_predictions, all_targets)
        val_maps.append(val_map)
        print(f"Validation mAP: {val_map:.4f}")

        # Check for improvement
        if val_map > best_map:
            best_map = val_map
            print(f"New best mAP: {best_map:.4f}")
            # Save best model
            torch.save(model.state_dict(), best_model_path)

    # Training complete
    total_time = time.time() - start_time
    print(f"Training complete in {total_time / 60:.2f} minutes")
    print(f"Best mAP: {best_map:.4f}")

    # Plot training results
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title(f'Training Loss ({model_type})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(val_maps)
    plt.title(f'Validation mAP ({model_type})')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')

    plt.tight_layout()
    plt.savefig(os.path.join(dataset_path, f'training_{model_type}.png'))

    return model, train_losses, val_maps


def evaluate_model(model, dataset_path, num_samples=5, confidence_threshold=0.5):
    """Evaluate model on sample images"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # Create dataset
    dataset = VOCInstanceSegmentationDataset(dataset_path, transforms=get_transform(train=False), train=False)

    # Select random samples
    indices = torch.randperm(len(dataset))[:num_samples].tolist()

    results = []

    with torch.no_grad():
        for idx in indices:
            # Get image and target
            img, target = dataset[idx]

            # Make prediction
            img_tensor = img.unsqueeze(0).to(device)
            prediction = model(img_tensor)[0]

            # Convert to CPU
            prediction = {k: v.cpu() for k, v in prediction.items()}

            # Convert image for visualization
            if isinstance(img, torch.Tensor):
                img_numpy = img.permute(1, 2, 0).numpy()
            else:
                img_numpy = np.array(img)

            # Filter predictions by confidence
            keep = prediction['scores'] >= confidence_threshold
            boxes = prediction['boxes'][keep]
            labels = prediction['labels'][keep]
            scores = prediction['scores'][keep]
            masks = prediction['masks'][keep]

            # Calculate IoUs
            ious = []
            for pred_box, pred_label in zip(boxes, labels):
                best_iou = 0
                for gt_box, gt_label in zip(target['boxes'], target['labels']):
                    if pred_label == gt_label:
                        iou = calculate_iou(pred_box, gt_box)
                        best_iou = max(best_iou, iou)
                ious.append(best_iou)

            # Store result
            results.append({
                'image': img_numpy,
                'pred_boxes': boxes.numpy(),
                'pred_labels': labels.numpy(),
                'pred_scores': scores.numpy(),
                'pred_masks': masks.squeeze(1).numpy(),
                'gt_boxes': target['boxes'].numpy(),
                'gt_labels': target['labels'].numpy(),
                'gt_masks': target['masks'].numpy() if 'masks' in target else np.array([]),
                'ious': ious
            })

    return results


def visualize_segmentation_results(results, output_path):
    """Visualize segmentation results"""
    # Create output directory
    os.makedirs(output_path, exist_ok=True)

    # Create figure
    plt.figure(figsize=(12, 6 * len(results)))

    for i, result in enumerate(results):
        # Ground truth visualization
        plt.subplot(len(results), 2, 2 * i + 1)
        plt.imshow(result['image'])
        plt.title('Ground Truth')
        plt.axis('off')

        # Draw ground truth boxes and masks
        for j, (box, label) in enumerate(zip(result['gt_boxes'], result['gt_labels'])):
            # Draw box
            x1, y1, x2, y2 = box
            plt.gca().add_patch(plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                fill=False, edgecolor='green', linewidth=2
            ))

            # Draw label
            class_name = TARGET_CLASSES[label - 1] if 0 <= label - 1 < len(TARGET_CLASSES) else f"Class {label}"
            plt.text(x1, y1, class_name, bbox=dict(facecolor='green', alpha=0.5))

            # Draw mask if available
            if j < len(result['gt_masks']):
                mask = result['gt_masks'][j]
                colored_mask = np.zeros_like(result['image'])
                colored_mask[mask > 0.5] = [0, 1.0, 0]  # Green mask
                plt.imshow(colored_mask, alpha=0.3)

        # Prediction visualization
        plt.subplot(len(results), 2, 2 * i + 2)
        plt.imshow(result['image'])
        plt.title('Predictions')
        plt.axis('off')

        # Draw predicted boxes and masks
        for j, (box, label, score, mask, iou) in enumerate(zip(
                result['pred_boxes'],
                result['pred_labels'],
                result['pred_scores'],
                result['pred_masks'],
                result['ious'])):

            # Draw box
            x1, y1, x2, y2 = box

            # Color based on IoU
            if iou >= 0.7:
                color = 'lime'
            elif iou >= 0.5:
                color = 'yellow'
            else:
                color = 'red'

            plt.gca().add_patch(plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                fill=False, edgecolor=color, linewidth=2
            ))

            # Draw label
            class_name = TARGET_CLASSES[label - 1] if 0 <= label - 1 < len(TARGET_CLASSES) else f"Class {label}"
            plt.text(x1, y1, f"{class_name}: {score:.2f}, IoU: {iou:.2f}",
                     bbox=dict(facecolor=color, alpha=0.5))

            # Draw mask
            mask_binary = mask > 0.5
            colored_mask = np.zeros_like(result['image'])

            # Set mask color based on IoU
            if iou >= 0.7:
                colored_mask[mask_binary] = [0, 1.0, 0]  # Green
            elif iou >= 0.5:
                colored_mask[mask_binary] = [1.0, 1.0, 0]  # Yellow
            else:
                colored_mask[mask_binary] = [1.0, 0, 0]  # Red

            plt.imshow(colored_mask, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'segmentation_results.png'))
    plt.close()

    print(f"Results saved to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description='Instance Segmentation using Mask R-CNN')
    parser.add_argument('--dataset_path', type=str, default='dataset_E4888', help='Path to dataset')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--eval_only', action='store_true', help='Only evaluate the model')
    parser.add_argument('--skip_scratch', action='store_true', help='Skip training from scratch')
    parser.add_argument('--skip_pretrained', action='store_true', help='Skip training with transfer learning')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Create output directories
    os.makedirs(args.dataset_path, exist_ok=True)
    os.makedirs(os.path.join(args.dataset_path, 'results'), exist_ok=True)

    # Train models unless evaluation only
    if not args.eval_only:
        # Train from scratch
        if not args.skip_scratch:
            print("\n===== TRAINING FROM SCRATCH =====")
            model_scratch, losses_scratch, maps_scratch = train_model(
                args.dataset_path, use_pretrained=False, num_epochs=args.epochs
            )

        # Train with transfer learning
        if not args.skip_pretrained:
            print("\n===== TRAINING WITH TRANSFER LEARNING =====")
            model_transfer, losses_transfer, maps_transfer = train_model(
                args.dataset_path, use_pretrained=True, num_epochs=args.epochs
            )

    # Evaluation
    print("\n===== EVALUATION =====")

    # Define models to evaluate
    models_to_evaluate = []

    # Check for transfer learning model
    transfer_model_path = os.path.join(args.dataset_path, 'best_transfer_segmentation_model.pth')
    if os.path.exists(transfer_model_path):
        models_to_evaluate.append(('transfer', transfer_model_path))

    # Check for scratch model
    scratch_model_path = os.path.join(args.dataset_path, 'best_scratch_segmentation_model.pth')
    if os.path.exists(scratch_model_path):
        models_to_evaluate.append(('scratch', scratch_model_path))

    # Evaluate each model
    for model_name, model_path in models_to_evaluate:
        print(f"\nEvaluating {model_name} model...")

        # Load model
        model = get_instance_segmentation_model(num_classes=len(TARGET_CLASSES) + 1)
        model.load_state_dict(torch.load(model_path))

        # Evaluate
        results = evaluate_model(model, args.dataset_path)

        # Visualize
        output_dir = os.path.join(args.dataset_path, f'results_{model_name}')
        os.makedirs(output_dir, exist_ok=True)
        visualize_segmentation_results(results, output_dir)