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
CLASS_MAPPING = {cls: idx for idx, cls in enumerate(TARGET_CLASSES)}


# Custom collate function to avoid pickling issues
def collate_fn(batch):
    return tuple(zip(*batch))


class VOCInstanceSegmentationDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms

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

        # Load mask
        mask_path = self.segmentations[idx]
        mask = Image.open(mask_path)
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

        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name in TARGET_CLASSES:
                # Get class index (0 or 1 for our two classes)
                label = CLASS_MAPPING[class_name]

                # Get bounding box
                bbox = obj.find('bndbox')
                xmin = float(bbox.find('xmin').text)
                ymin = float(bbox.find('ymin').text)
                xmax = float(bbox.find('xmax').text)
                ymax = float(bbox.find('ymax').text)

                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(label)

        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

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

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.images)


def get_transform():
    transforms = []
    transforms.append(torchvision.transforms.ToTensor())
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


def train_model(dataset_path, use_pretrained=True, num_epochs=5):
    # Set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    # Create dataset and dataloader
    dataset = VOCInstanceSegmentationDataset(dataset_path, get_transform())
    print(f"Dataset size: {len(dataset)} images")

    # Split dataset into train and validation (80/20 split)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    indices = torch.randperm(len(dataset)).tolist()
    dataset_train = torch.utils.data.Subset(dataset, indices[:train_size])
    dataset_val = torch.utils.data.Subset(dataset, indices[train_size:])

    print(f"Training set: {len(dataset_train)} images")
    print(f"Validation set: {len(dataset_val)} images")

    # Dataloaders - FIXED by using a named function instead of lambda
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

    # Create model - for our one-stage implementation, we're using Mask R-CNN
    # While Mask R-CNN is technically two-stage, we're using it as a representative model
    # In practice, a true one-stage instance segmentation model like YOLACT would be used
    num_classes = len(TARGET_CLASSES) + 1  # +1 for background

    # Get model
    model = get_instance_segmentation_model(num_classes, pretrained=use_pretrained)

    # Move model to device
    model.to(device)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Training loop
    print("Starting training...")
    start_time = time.time()

    # Track metrics
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        # Training
        model.train()
        epoch_loss = 0

        for images, targets in tqdm(data_loader_train):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # Backward pass
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()

        # Update learning rate
        lr_scheduler.step()

        # Calculate average loss
        avg_train_loss = epoch_loss / len(data_loader_train)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for images, targets in tqdm(data_loader_val):
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                # Forward pass
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                val_loss += losses.item()

        # Calculate average validation loss
        avg_val_loss = val_loss / len(data_loader_val)
        val_losses.append(avg_val_loss)

        print(f"Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

    total_time = time.time() - start_time
    print(f"Training complete in {total_time / 60:.2f} minutes")

    # Save model
    torch.save(model.state_dict(), os.path.join(dataset_path, 'instance_segmentation_model.pth'))

    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(dataset_path, 'segmentation_training_loss.png'))

    return model, train_losses, val_losses


def evaluate_model(model, dataset_path, num_samples=5):
    """Evaluate model on sample images from the dataset"""
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.eval()

    # Create dataset
    dataset = VOCInstanceSegmentationDataset(dataset_path, get_transform())

    # Get sample indices
    indices = torch.randperm(len(dataset))[:num_samples].tolist()

    results = []

    for idx in indices:
        # Get image and target
        img, target = dataset[idx]

        # Make prediction
        with torch.no_grad():
            prediction = model([img.to(device)])[0]

        # Convert image to numpy for visualization
        img = img.permute(1, 2, 0).cpu().numpy()

        # Get masks, boxes, labels and scores
        pred_masks = prediction['masks'].cpu().numpy()
        pred_boxes = prediction['boxes'].cpu().numpy()
        pred_labels = prediction['labels'].cpu().numpy()
        pred_scores = prediction['scores'].cpu().numpy()

        # Get ground truth masks, boxes and labels
        gt_masks = target['masks'].cpu().numpy()
        gt_boxes = target['boxes'].cpu().numpy()
        gt_labels = target['labels'].cpu().numpy()

        results.append({
            'image': img,
            'pred_masks': pred_masks,
            'pred_boxes': pred_boxes,
            'pred_labels': pred_labels,
            'pred_scores': pred_scores,
            'gt_masks': gt_masks,
            'gt_boxes': gt_boxes,
            'gt_labels': gt_labels
        })

    return results


def visualize_segmentation_results(results, dataset_path):
    """Visualize instance segmentation results"""
    class_names = TARGET_CLASSES

    for i, result in enumerate(results):
        plt.figure(figsize=(12, 6))

        # Plot ground truth
        plt.subplot(1, 2, 1)
        plt.imshow(result['image'])
        plt.title('Ground Truth')

        # Apply masks with different colors
        img_gt = result['image'].copy()
        for j, (mask, label) in enumerate(zip(result['gt_masks'], result['gt_labels'])):
            color = np.array([0, 1.0, 0, 0.5])  # Green with alpha
            mask_img = mask[0, :, :, None] * color.reshape(1, 1, 4)
            img_gt = np.where(mask_img, mask_img, img_gt)

            # Also draw bounding box
            box = result['gt_boxes'][j]
            x1, y1, x2, y2 = box
            plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                              fill=False, edgecolor='green', linewidth=2))
            plt.text(x1, y1, class_names[label],
                     bbox=dict(facecolor='green', alpha=0.5))

        plt.imshow(img_gt)

        # Plot predictions
        plt.subplot(1, 2, 2)
        plt.imshow(result['image'])
        plt.title('Predictions')

        # Apply masks with different colors for predictions
        img_pred = result['image'].copy()
        for j, (mask, label, score) in enumerate(zip(result['pred_masks'],
                                                     result['pred_labels'],
                                                     result['pred_scores'])):
            if score > 0.5:  # Only show predictions with confidence > 0.5
                color = np.array([1.0, 0, 0, 0.5])  # Red with alpha
                # Masks have shape [N, 1, H, W]
                mask_img = mask[0, :, :, None] * color.reshape(1, 1, 4)
                img_pred = np.where(mask_img, mask_img, img_pred)

                # Also draw bounding box
                box = result['pred_boxes'][j]
                x1, y1, x2, y2 = box
                plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                                  fill=False, edgecolor='red', linewidth=2))
                plt.text(x1, y1, f"{class_names[label - 1]}: {score:.2f}",
                         bbox=dict(facecolor='red', alpha=0.5))

        plt.imshow(img_pred)

        plt.savefig(os.path.join(dataset_path, f'segmentation_result_{i}.png'))
        plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description='One-Stage Instance Segmentation')
    parser.add_argument('--dataset_path', type=str, default='dataset_E4888', help='Path to the dataset')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training')
    parser.add_argument('--skip_scratch', action='store_true', help='Skip training from scratch')
    parser.add_argument('--skip_pretrained', action='store_true', help='Skip training with pretrained weights')
    parser.add_argument('--eval_only', action='store_true', help='Only evaluate the model')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Create output directories if they don't exist
    os.makedirs(args.dataset_path, exist_ok=True)
    os.makedirs(os.path.join(args.dataset_path, 'models'), exist_ok=True)
    os.makedirs(os.path.join(args.dataset_path, 'results'), exist_ok=True)

    # Training and evaluation
    if not args.eval_only:
        # Train model from scratch
        if not args.skip_scratch:
            print("Training model from scratch...")
            model_scratch, losses_scratch, val_losses_scratch = train_model(
                args.dataset_path, use_pretrained=False, num_epochs=args.epochs)

        # Train model with pre-trained weights
        if not args.skip_pretrained:
            print("Training model with pre-trained weights...")
            model_pretrained, losses_pretrained, val_losses_pretrained = train_model(
                args.dataset_path, use_pretrained=True, num_epochs=args.epochs)

            # Compare training losses if both models were trained
            if not args.skip_scratch:
                plt.figure(figsize=(12, 6))
                plt.subplot(1, 2, 1)
                plt.plot(losses_scratch, label='From Scratch')
                plt.plot(losses_pretrained, label='Pre-trained')
                plt.title('Training Loss Comparison')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()

                plt.subplot(1, 2, 2)
                plt.plot(val_losses_scratch, label='From Scratch')
                plt.plot(val_losses_pretrained, label='Pre-trained')
                plt.title('Validation Loss Comparison')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()

                plt.savefig(os.path.join(args.dataset_path, 'segmentation_loss_comparison.png'))

    # Evaluate model
    print("Evaluating pre-trained model...")
    # Load model
    model = get_instance_segmentation_model(num_classes=len(TARGET_CLASSES) + 1)
    model_path = os.path.join(args.dataset_path, 'instance_segmentation_model.pth')

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        results = evaluate_model(model, args.dataset_path)
        visualize_segmentation_results(results, args.dataset_path)
    else:
        print(f"Model file not found: {model_path}")
        if args.eval_only:
            print("Please train the model first or provide a valid model path.")