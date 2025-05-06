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
CLASS_MAPPING = {cls: idx for idx, cls in enumerate(TARGET_CLASSES)}


# Custom collate function to avoid pickling issues
def collate_fn(batch):
    return tuple(zip(*batch))


class CustomVOCDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms

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

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.images)


def get_transform():
    transforms = []
    transforms.append(torchvision.transforms.ToTensor())
    return torchvision.transforms.Compose(transforms)


def train_model(dataset_path, use_pretrained=True, num_epochs=5, subset_size=1.0):
    # Set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")


    # Create dataset and dataloader
    dataset = CustomVOCDataset(dataset_path, get_transform())
    full_size = len(dataset)
    print(f"Full dataset size: {full_size} images")

    # Apply subset size if less than 1.0
    if subset_size < 1.0:
        subset_size = max(0.01, min(1.0, subset_size))  # Ensure it's between 0.01 and 1.0
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
        batch_size=2,
        shuffle=False,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        collate_fn=collate_fn
    )

    # Create model - two-stage Faster R-CNN with ResNet50 backbone
    num_classes = len(TARGET_CLASSES) + 1  # +1 for internal background handling

    if use_pretrained:
        print("Loading pre-trained model...")
        # Load pre-trained model
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn(weights=weights)

        # Replace the classifier with a new one for our classes
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    else:
        print("Training model from scratch...")
        # Create model from scratch
        model = fasterrcnn_resnet50_fpn(weights=None, num_classes=num_classes)

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

            # FIXED: Handle both list and dictionary return types
            if isinstance(loss_dict, dict):
                losses = sum(loss for loss in loss_dict.values())
            elif isinstance(loss_dict, list):
                # Handle list of losses carefully
                total_loss = 0
                for loss in loss_dict:
                    if isinstance(loss, dict):
                        # If an element is a dictionary, sum its values
                        total_loss += sum(v for v in loss.values())
                    else:
                        # If it's a number or tensor, add it directly
                        total_loss += loss
                losses = total_loss
            else:
                # If it's a single loss value
                losses = loss_dict

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

                # FIXED: Handle both list and dictionary return types
                if isinstance(loss_dict, dict):
                    losses = sum(loss for loss in loss_dict.values())
                else:
                    # If it's a list or tensor, sum directly
                    losses = sum(loss_dict) if isinstance(loss_dict, list) else loss_dict

                val_loss += losses.item()

        # Calculate average validation loss
        avg_val_loss = val_loss / len(data_loader_val)
        val_losses.append(avg_val_loss)

        print(f"Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

    total_time = time.time() - start_time
    print(f"Training complete in {total_time / 60:.2f} minutes")

    # Save model
    torch.save(model.state_dict(), os.path.join(dataset_path, 'faster_rcnn_model.pth'))

    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(dataset_path, 'training_loss.png'))

    return model, train_losses, val_losses


# Custom head for the model
class FastRCNNPredictor(torch.nn.Module):
    def __init__(self, in_channels, num_classes):
        super(FastRCNNPredictor, self).__init__()
        self.cls_score = torch.nn.Linear(in_channels, num_classes)
        self.bbox_pred = torch.nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas


def evaluate_model(model, dataset_path, num_samples=5):
    """Evaluate model on sample images from the dataset"""
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.eval()

    # Create dataset
    dataset = CustomVOCDataset(dataset_path, get_transform())

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

        # Get boxes, labels and scores
        pred_boxes = prediction['boxes'].cpu().numpy()
        pred_labels = prediction['labels'].cpu().numpy()
        pred_scores = prediction['scores'].cpu().numpy()

        # Get ground truth boxes and labels
        gt_boxes = target['boxes'].cpu().numpy()
        gt_labels = target['labels'].cpu().numpy()

        results.append({
            'image': img,
            'pred_boxes': pred_boxes,
            'pred_labels': pred_labels,
            'pred_scores': pred_scores,
            'gt_boxes': gt_boxes,
            'gt_labels': gt_labels
        })

    return results


def visualize_results(results, dataset_path):
    """Visualize detection results"""
    class_names = TARGET_CLASSES

    for i, result in enumerate(results):
        plt.figure(figsize=(12, 6))

        # Plot ground truth
        plt.subplot(1, 2, 1)
        plt.imshow(result['image'])
        plt.title('Ground Truth')

        for box, label in zip(result['gt_boxes'], result['gt_labels']):
            x1, y1, x2, y2 = box
            plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                              fill=False, edgecolor='green', linewidth=2))
            plt.text(x1, y1, class_names[label],
                     bbox=dict(facecolor='green', alpha=0.5))

        # Plot predictions
        plt.subplot(1, 2, 2)
        plt.imshow(result['image'])
        plt.title('Predictions')

        for box, label, score in zip(result['pred_boxes'], result['pred_labels'], result['pred_scores']):
            if score > 0.5:  # Only show predictions with confidence > 0.5
                x1, y1, x2, y2 = box
                plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                                  fill=False, edgecolor='red', linewidth=2))
                plt.text(x1, y1, f"{class_names[label - 1]}: {score:.2f}",
                         bbox=dict(facecolor='red', alpha=0.5))

        plt.savefig(os.path.join(dataset_path, f'result_{i}.png'))
        plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Two-Stage Object Detection using Faster R-CNN')
    parser.add_argument('--dataset_path', type=str, default='dataset_E4888', help='Path to the dataset')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training')
    parser.add_argument('--skip_scratch', action='store_true', help='Skip training from scratch')
    parser.add_argument('--skip_pretrained', action='store_true', help='Skip training with pretrained weights')
    parser.add_argument('--eval_only', action='store_true', help='Only evaluate the model')
    parser.add_argument('--subset_size', type=float, default=1.0,
                        help='Fraction of dataset to use (0.0-1.0). Use 0.5 for 50% of images.')
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
                args.dataset_path, use_pretrained=False, num_epochs=args.epochs, subset_size=args.subset_size)

        # Train model with pre-trained weights
        if not args.skip_pretrained:
            print("Training model with pre-trained weights...")
            model_pretrained, losses_pretrained, val_losses_pretrained = train_model(
                args.dataset_path, use_pretrained=True, num_epochs=args.epochs, subset_size=args.subset_size)

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

                plt.savefig(os.path.join(args.dataset_path, 'loss_comparison.png'))

    # Evaluate model
    print("Evaluating pre-trained model...")
    # Load model
    model = fasterrcnn_resnet50_fpn(num_classes=len(TARGET_CLASSES) + 1)
    model_path = os.path.join(args.dataset_path, 'faster_rcnn_model.pth')

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        results = evaluate_model(model, args.dataset_path)
        visualize_results(results, args.dataset_path)
    else:
        print(f"Model file not found: {model_path}")
        if args.eval_only:
            print("Please train the model first or provide a valid model path.")