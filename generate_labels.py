from pathlib import Path
from ultralytics import YOLO
from ultralytics.data.split import autosplit
import yaml

# The initial diresctory structure should look like this:
# 
# dataset_dir
# |--- images
# |    |--- image1.jpg
# |    |--- image2.jpg
# |    |--- image3.jpg


dataset_dir = Path("./datasets/actual-hoop")
model_weights = Path("./runs/detect/yolo11-nano-actual-hoop-1000-images/weights/best.pt")

# Load model and predict bounding boxes
model = YOLO(model_weights)
results = model.predict(source=dataset_dir/'images', conf=0.5)


# Delete any existing image paths
if (dataset_dir/'all_images.txt').exists():
    (dataset_dir/'all_images.txt').unlink()
    (dataset_dir/'all_images.txt').touch()

# Delete any existing predictions
if (dataset_dir/'predictions').exists():
    for file in (dataset_dir/'predictions').iterdir():
       file.unlink()

(dataset_dir/'predictions').mkdir()


with open(dataset_dir/'all_images.txt', 'a') as f:
    for result in results:
        orig_img_path = Path(result.path)

        label_filepath = dataset_dir/'labels'/(orig_img_path.stem + ".txt")
        if not len(result) <= 0:
            result[0].save_txt(txt_file=label_filepath)
            result[0].save(dataset_dir/"predictions"/(orig_img_path.stem + "_annotated" + orig_img_path.suffix))
        f.write("./" + str(label_filepath.relative_to(dataset_dir)) + "\n")

# Autosplit the dataset into train, validation & test sets
autosplit(
    path=dataset_dir/'images',
    weights=(0.9, 0.1, 0)
)

# Create a data.yaml file for the dataset
dataset_yaml = {
    'path': str(dataset_dir),
    'names': results[0].names,
    'train': str(dataset_dir.relative_to(dataset_dir)/'autosplit_train.txt'),
    'val': str(dataset_dir.relative_to(dataset_dir)/'autosplit_val.txt')
}

# If a test set exists, add it to the yaml file
if (dataset_dir.relative_to(dataset_dir)/'autosplit_test.txt').exists():
    dataset_yaml['test'] = str(dataset_dir.relative_to(dataset_dir)/'autosplit_test.txt')

yaml.dump(dataset_yaml, open(dataset_dir/'data.yaml', 'w'))