from pathlib import Path
from ultralytics import YOLO
import argparse
from ultralytics.data.split import autosplit
from generate_yaml import generate_yaml

# The initial directory structure should look like this:
# 
# dataset_dir
# |--- images
# |    |--- image1.jpg
# |    |--- image2.jpg
# |    |--- image3.jpg

def generate_labels(dataset_dir: Path, model_weights: Path, confidence: int = 0.5, save_predictions: bool = False, autosplit_dataset: bool = False, autosplit_weights: tuple = (0.9, 0.1, 0)):
    """
    Generate labels for images in the dataset directory using a YOLO model.
    
    Args:
        dataset_dir (Path): Path to the dataset directory containing images.
        model_weights (Path): Path to the YOLO model weights file.
    """
    
    dataset_dir = dataset_dir.absolute()

    # Load model and predict bounding boxes
    model = YOLO(model_weights)
    results = model.predict(source=dataset_dir/'images', stream=True, conf=0.5)


    # Delete any existing image paths
    if (dataset_dir/'all_images.txt').exists():
        (dataset_dir/'all_images.txt').unlink()
        (dataset_dir/'all_images.txt').touch()

    # Delete any existing predictions
    if save_predictions:
        if (dataset_dir/'predictions').exists():
            for file in (dataset_dir/'predictions').iterdir():
                file.unlink()
            (dataset_dir/'predictions').rmdir()

        (dataset_dir/'predictions').mkdir()

    # Delete any existing labels
    if (dataset_dir/'labels').exists():
        for file in (dataset_dir/'labels').iterdir():
            file.unlink()
        (dataset_dir/'labels').rmdir()

    (dataset_dir/'labels').mkdir()


    with open(dataset_dir/'all_images.txt', 'a') as f:
        for result in results:
            orig_img_path = Path(result.path)

            label_filepath = dataset_dir/'labels'/(orig_img_path.stem + ".txt")

            if len(result.boxes) > 0:
                result.save_txt(txt_file=label_filepath)
                if save_predictions:
                    result.save(dataset_dir/"predictions"/(orig_img_path.stem + "_annotated" + orig_img_path.suffix))
            
            f.write("./" + str(orig_img_path.relative_to(dataset_dir)) + "\n")

    # Autosplit the dataset into train, validation & test sets
    if autosplit_dataset:
        autosplit(
            path=dataset_dir/'images',
            weights=autosplit_weights
        )

    generate_yaml(dataset_dir, model, is_autosplit=autosplit_dataset)

if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Generate labels for images in a dataset directory using a YOLO model.")
    parser.add_argument('dataset_dir', type=Path, help='Path to the dataset directory containing images.')
    parser.add_argument('model_weights', type=Path, help='Path to the YOLO model weights file.')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold for predictions.')
    parser.add_argument('--save', action='store_true', help='Save prediction images with bounding boxes.')
    parser.add_argument('--autosplit', action='store_true', help='Automatically split the dataset into train, validation, and test sets.')
    parser.add_argument('--split', type=float, nargs=3, metavar=('TRAIN', 'VAL', 'TEST'), default=(0.9, 0.1, 0), help='Weights for autosplitting the dataset (train, val, test).')
    
    args = parser.parse_args()

    generate_labels(args.dataset_dir, args.model_weights, args.conf, args.save, args.autosplit, tuple(args.split))