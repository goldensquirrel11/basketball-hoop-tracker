from pathlib import Path
from zipfile import ZipFile
import yaml
import argparse

def yolo_to_cvat(dataset_dir: Path, output_dir: Path):
    """
    Convert YOLO format labels to CVAT format.
    
    Args:
        dataset_dir (Path): Path to the dataset directory containing 'labels' folder.
    """
    
    # Input dataset details
    images_dir = dataset_dir / 'images'
    labels_dir = dataset_dir / 'labels'
    all_images_txt = dataset_dir / 'all_images.txt'
    dataset_yaml = dataset_dir / 'data.yaml'

    # Output dataset details
    image_zipfile_dir = output_dir / f'{dataset_dir.name}_images.zip'
    dataset_zipfile_dir = output_dir / f'{dataset_dir.name}_dataset.zip'
    train_txt = output_dir / 'train.txt'
    output_data_yaml = output_dir / 'data.yaml'

    # Remove existing temp files
    if train_txt.exists():
        train_txt.unlink()
    if output_data_yaml.exists():
        output_data_yaml.unlink()
    
    # Create output data.yaml
    with open(dataset_yaml, 'r') as f:
        dataset_yaml = {}
        
        dataset_yaml['names'] = yaml.safe_load(f)['names']
        dataset_yaml['path'] = '.'
        dataset_yaml['train'] = 'train.txt'
    
        yaml.dump(dataset_yaml, open(output_data_yaml, 'w'))

    # Create zip file containing only images
    with ZipFile(image_zipfile_dir, 'w') as image_zip:
        for img_file in images_dir.glob('*.*'):
            image_zip.write(img_file, img_file.name)

    # Create zip file containing the complete dataset
    with ZipFile(dataset_zipfile_dir, 'w') as dataset_zip:
        # Add images
        for img_file in images_dir.glob('*.*'):
            dataset_zip.write(img_file, f'images/train/{img_file.name}')

        # Add labels
        for label_file in labels_dir.glob('*.txt'):
            dataset_zip.write(label_file, f'labels/train/{label_file.name}')

        # Add train.txt
        with open(all_images_txt, 'r') as f:
            with open(train_txt, 'a') as t:
                for line in f:
                    img = Path(line)    # Remove './' prefix
                    t.write(f'data/images/train/{img.name}')

                dataset_zip.write(train_txt, 'train.txt')
                train_txt.unlink()
        
        # Add data.yaml
        dataset_zip.write(output_data_yaml, 'data.yaml')
        output_data_yaml.unlink()

if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser(description="Convert YOLO 11 dataset to CVAT compliant dataset zipfiles.")
    parser.add_argument('dataset_dir', type=Path, help='Path to the YOLO dataset directory.')
    parser.add_argument('output_dir', type=Path, help='Path to the output directory for CVAT dataset zipfiles.')
    args = parser.parse_args()

    yolo_to_cvat(args.dataset_dir, args.output_dir)
    print(f"Converted {args.dataset_dir.name} to CVAT compliant dataset in {args.output_dir.name}.")