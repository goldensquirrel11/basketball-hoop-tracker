import os
from pathlib import Path
import argparse
import json

def delete_similar_images(images_dir: Path):
    similar_images = True
    total_deleted_images = 0
    
    while similar_images:
        print()
        os.system(f'./bin/linux_czkawka_cli image -z Lanczos3 -d {images_dir} -p similar-images.json --delete-method AES --do-not-print-results --similarity-preset Minimal')

        output_json_file = Path.cwd() / 'similar-images.json'

        with open(output_json_file, 'r') as f:
            similar_images = json.load(f)
            total_deleted_images = total_deleted_images + len(similar_images)
        
        if not similar_images:
            print("No more similar images found.")
            output_json_file.unlink()
        else:
            print(f"Deleted {len(similar_images)} similar images")

    print(f"\nDeleted {total_deleted_images} total similar images")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Delete similar images in a directory.")
    parser.add_argument('images_dir', type=Path, help='Directory containing images to check for similarity.')
    args = parser.parse_args()

    if not args.images_dir.exists():
        print(f"Directory {args.images_dir} does not exist.")
        exit(1)

    delete_similar_images(args.images_dir)