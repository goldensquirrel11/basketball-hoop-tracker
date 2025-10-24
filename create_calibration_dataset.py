from pathlib import Path
import random
from time import sleep
import shutil
import argparse

def create_calibration_dataset(dataset_dir: Path, calibration_dataset_dir: Path, num_of_calib_images = 100):
    image_files = list((dataset_dir / 'images').glob('*'))

    # Check if there are enough images to copy
    if len(image_files) > num_of_calib_images:
        print(f"Found {len(image_files)} images, Copying {num_of_calib_images}.")

        sleep(1)  # Optional: wait for a second before proceeding

        # Randomly select images to copy
        images_to_copy = random.sample(image_files, num_of_calib_images)
        
        if (not calibration_dataset_dir.exists()):
            calibration_dataset_dir.mkdir()

        # Copy the selected images
        for image in images_to_copy:
            shutil.copy(image, calibration_dataset_dir/image.name)
            print(f"copied: {image}")

if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser(description="Extract a random subset of images from a dataset to be used as a quantization calibration dataset")
    parser.add_argument('dataset_dir', type=Path, help='Dataset directory.')
    parser.add_argument('output_dir', type=Path, help='Directory to save the calibration dataeset images.')
    parser.add_argument('num_images', type=int, help='Number of images to sample from the original dataset')
    args = parser.parse_args()

    create_calibration_dataset(args.dataset_dir, args.output_dir, args.num_images)