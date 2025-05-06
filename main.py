import os
import argparse
import time


def main():
    parser = argparse.ArgumentParser(description='E4888 Lab Assignment - Object Detection and Instance Segmentation')
    parser.add_argument('--dataset_path', type=str, default='dataset_E4888', help='Path to dataset')
    parser.add_argument('--prepare_data', action='store_true', help='Prepare dataset')
    parser.add_argument('--analyze_data', action='store_true', help='Analyze dataset')
    parser.add_argument('--train_detection', action='store_true', help='Train object detection model')
    parser.add_argument('--train_segmentation', action='store_true', help='Train instance segmentation model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate models')
    parser.add_argument('--all', action='store_true', help='Run all steps')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.dataset_path, exist_ok=True)

    # Define run modes
    run_preparation = args.prepare_data or args.all
    run_analysis = args.analyze_data or args.all
    run_detection = args.train_detection or args.all
    run_segmentation = args.train_segmentation or args.all
    run_evaluation = args.evaluate or args.all

    # Step 1: Data Preparation
    if run_preparation:
        print("=== Step 1: Data Preparation ===")
        from data_preparation import process_dataset
        process_dataset()

    # Step 2: Data Analysis
    if run_analysis:
        print("\n=== Step 2: Data Analysis ===")
        from data_analysis import analyze_dataset
        analyze_dataset(args.dataset_path)

    # Step 3: Object Detection Training
    if run_detection:
        print("\n=== Step 3: Object Detection Training (Two-Stage) ===")
        from object_detection import train_model as train_detection

        # Train from scratch
        print("Training object detection model from scratch...")
        start_time = time.time()
        model_scratch, losses_scratch, val_losses_scratch = train_detection(
            args.dataset_path, use_pretrained=False, num_epochs=5)
        scratch_time = time.time() - start_time

        # Train with pre-trained weights
        print("Training object detection model with pre-trained weights...")
        start_time = time.time()
        model_pretrained, losses_pretrained, val_losses_pretrained = train_detection(
            args.dataset_path, use_pretrained=True, num_epochs=5)
        pretrained_time = time.time() - start_time

        print(f"Training time from scratch: {scratch_time / 60:.2f} minutes")
        print(f"Training time with pre-trained weights: {pretrained_time / 60:.2f} minutes")

    # Step 4: Instance Segmentation Training
    if run_segmentation:
        print("\n=== Step 4: Instance Segmentation Training (One-Stage) ===")
        from instance_segmentation import train_model as train_segmentation

        # Train from scratch
        print("Training instance segmentation model from scratch...")
        start_time = time.time()
        model_scratch, losses_scratch, val_losses_scratch = train_segmentation(
            args.dataset_path, use_pretrained=False, num_epochs=5)
        scratch_time = time.time() - start_time

        # Train with pre-trained weights
        print("Training instance segmentation model with pre-trained weights...")
        start_time = time.time()
        model_pretrained, losses_pretrained, val_losses_pretrained = train_segmentation(
            args.dataset_path, use_pretrained=True, num_epochs=5)
        pretrained_time = time.time() - start_time

        print(f"Training time from scratch: {scratch_time / 60:.2f} minutes")
        print(f"Training time with pre-trained weights: {pretrained_time / 60:.2f} minutes")

    # Step 5: Evaluation
    if run_evaluation:
        print("\n=== Step 5: Evaluation ===")

        if run_detection:
            from object_detection import evaluate_model as evaluate_detection, visualize_results

            print("Evaluating object detection model...")
            # Load model
            from torchvision.models.detection import fasterrcnn_resnet50_fpn
            import torch

            model = fasterrcnn_resnet50_fpn(num_classes=3)  # 2 classes + background
            model.load_state_dict(torch.load(os.path.join(args.dataset_path, 'faster_rcnn_model.pth')))

            results = evaluate_detection(model, args.dataset_path)
            visualize_results(results, args.dataset_path)

        if run_segmentation:
            from instance_segmentation import evaluate_model as evaluate_segmentation, visualize_segmentation_results

            print("Evaluating instance segmentation model...")
            # Load model
            from instance_segmentation import get_instance_segmentation_model
            import torch

            model = get_instance_segmentation_model(num_classes=3)  # 2 classes + background
            model.load_state_dict(torch.load(os.path.join(args.dataset_path, 'instance_segmentation_model.pth')))

            results = evaluate_segmentation(model, args.dataset_path)
            visualize_segmentation_results(results, args.dataset_path)

    print("\nAll tasks completed successfully!")
    print("Results and visualizations saved to:", args.dataset_path)


if __name__ == "__main__":
    main()