from pathlib import Path
import yaml
import pandas as pd
from collections import Counter
import random
from sklearn.model_selection import KFold
import shutil
from tqdm import tqdm
import argparse

def generate_kfolds(dataset_path: Path, k=5, random_seed=42):
    random.seed(random_seed)

    # retrieve dataset image names
    images = sorted((dataset_path / 'images').glob('*.*'))

    # read class labels
    yaml_file = dataset_path / 'data.yaml'

    with open(yaml_file, encoding='utf8') as y:
        classes = yaml.safe_load(y)['names']
    class_ids = sorted(classes.keys())
    indexes = [image.stem for image in images]  # uses base filename as ID (no file extension)
    
    # Create dataframe where the rows are indexed by filename and columns are indexed by class ID
    images_df = pd.DataFrame([], columns=class_ids, index=indexes)

    for image in tqdm(images, total=len(images), desc='Searching labels'):
        label = Path(dataset_path / 'labels' / f'{image.stem}.txt')
        
        if not label.exists():
            continue
        
        lbl_counter = Counter()

        with open(label) as label_file:
            lines = label_file.readlines()

        for line in lines:
            # classes for YOLO labels uses integer at first position of each line
            lbl_counter[int(line.split(' ', 1)[0])] += 1

        images_df.loc[image.stem] = lbl_counter

    images_df = images_df.infer_objects(copy=False).fillna(0.0)  # replace `nan` values with `0.0


    kf = KFold(n_splits=k, shuffle=True, random_state=20)  # setting random_state for repeatable results

    kfolds = list(kf.split(images_df))

    folds = [f'split_{n}' for n in range(1, k + 1)]
    folds_df = pd.DataFrame(index=indexes, columns=folds)

    for i, (train, val) in enumerate(kfolds, start=1):
        folds_df.loc[images_df.iloc[train].index, f'split_{i}'] = 'train'
        folds_df.loc[images_df.iloc[val].index, f'split_{i}'] = 'val'

    print('Image splits')
    print(folds_df)

    fold_label_distribution = pd.DataFrame(index=folds, columns=class_ids)

    for n, (train_indices, val_indices) in enumerate(kfolds, start=1):
        train_totals = images_df.iloc[train_indices].sum()
        val_totals = images_df.iloc[val_indices].sum()

        val_ratio = val_totals / (train_totals + val_totals) # Ratio of validation to total images
        fold_label_distribution.loc[f'split_{n}'] = val_ratio

    print('\n\nproportion of val images according to class ID and split')
    print(fold_label_distribution)


    # Create the necessary directories and dataset YAML files
    save_path = Path(dataset_path / f'{k}-Fold_Cross-val')
    save_path.mkdir(parents=True, exist_ok=True)
    split_dataset_yaml = []

    for split in folds_df.columns:
        # Create directories
        split_dir = save_path / split
        split_dir.mkdir(parents=True, exist_ok=True)

        
        (split_dir / 'images' / 'train' ).mkdir(parents=True, exist_ok=True)
        (split_dir / 'labels' / 'train').mkdir(parents=True, exist_ok=True)

        (split_dir / 'images' / 'val').mkdir(parents=True, exist_ok=True)
        (split_dir / 'labels' / 'val').mkdir(parents=True, exist_ok=True)

        # Create dataset YAML files
        split_dataset_yaml = split_dir / 'data.yaml'

        with open(split_dataset_yaml, 'w') as new_dataset_yaml:
            yaml.safe_dump(
                {
                    'path': split_dir.absolute().as_posix(),
                    'train': 'images/train',
                    'val': 'images/val',
                    'names': classes,
                },
                new_dataset_yaml
            )

    # Copy image & label files to the new split directories
    for image in tqdm(images, total=len(images), desc='Copying files'):
        for split, k_split in folds_df.loc[image.stem].items():
            label = dataset_path / 'labels' / f'{image.stem}.txt'

            # Destination directory
            img_to_path = save_path / split / 'images' / k_split
            lbl_to_path = save_path / split / 'labels' / k_split

            # Copy image and label files to new directory (SamefileError if file already exists)
            shutil.copy(image, img_to_path / image.name)
            if label.exists():
                shutil.copy(label, lbl_to_path / label.name)


    folds_df.to_csv(save_path / 'kfold_datasplit.csv')
    fold_label_distribution.to_csv(save_path / 'kfold_label_distribution.csv')

if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser(description="Generate k-folds of a dataset")
    parser.add_argument('dataset_dir', type=Path, help='Directory of the original dataset')
    parser.add_argument('-k', type=int, default=5, help='The number of folds (default = 5)')
    parser.add_argument('-s', '--seed', type=int, default=42, help='random seed (default = 42)')
    args = parser.parse_args()

    generate_kfolds(args.dataset_dir, args.k, args.seed)