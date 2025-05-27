from pathlib import Path
from ultralytics import YOLO
import yaml

def generate_yaml(dataset_dir: Path, model: YOLO, is_autosplit: bool = False):
    
    dataset_yaml = {}

    dataset_yaml['names'] = model.names
    dataset_yaml['path'] = str(dataset_dir)
    
    train_dir = dataset_dir/'autosplit_train.txt'
    val_dir = dataset_dir/'autosplit_val.txt'
    test_dir = dataset_dir/'autosplit_test.txt'

    if (is_autosplit):
        if not (train_dir.exists() and val_dir.exists()):
            raise FileNotFoundError("The dataset directory does not contain 'autosplit_train.txt' and 'autosplit_val.txt'. Please generate them first.")
        
        dataset_yaml['train'] = str(train_dir.relative_to(dataset_dir))
        dataset_yaml['val'] = str(val_dir.relative_to(dataset_dir))
    else:
        if not (dataset_dir / 'all_images.txt').exists():
            raise FileNotFoundError("The dataset directory does not contain 'all_images.txt'. Please generate it first.")

        dataset_yaml['train'] = str((dataset_dir / 'all_images.txt').relative_to(dataset_dir))
        dataset_yaml['val'] = str((dataset_dir / 'all_images.txt').relative_to(dataset_dir))

    if (test_dir.exists()):
        dataset_yaml['test'] = str(test_dir.relative_to(dataset_dir))

    yaml.dump(dataset_yaml, open(dataset_dir/'data.yaml', 'w'))

    print(f"Generated data.yaml for {dataset_dir.name}")