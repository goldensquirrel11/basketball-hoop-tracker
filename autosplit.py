from ultralytics.data.split import autosplit
import argparse
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autosplit dataset into train, val, and test sets.")
    parser.add_argument('dataset_path', type=str, help='Path to the dataset directory containing images.')
    parser.add_argument('weights', type=float, nargs=3, default=(0.8, 0.1, 0.1), help='Weights for train, val, and test splits.')
    
    args = parser.parse_args()
    
    print(f"Autosplitting dataset at {args.dataset_path} with weights {args.weights}...")

    autosplit(
        path=Path(args.dataset_path) / 'images',
        weights=args.weights
    )