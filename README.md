# Yolo Training Utils

A collection of scripts and utilities for dataset preparation, conversion, auto-annotation & training of YOLO object detection models.

## Utils/Scripts

**Dataset preparation:**
   - Place raw videos in the `videos/` directory.
   - Use `extract_frames.py` to extract all video frames.

**Generate labels:**
   - Use `generate_labels.py` to predict and save YOLO labels for your dataset.

**Split dataset:**
   - Use `autosplit.py` or `generate_kfold_dataset.py` for train/val/test splits.

**Train a model:**
   - Use `training.py` for automated queue-based training.

**Predict on videos:**
   - Use `predict_video.py` or `batch_predict_video.py` to run inference and save annotated videos.

**Format conversion:**
   - If you use CVAT for annotation, use `yolo_to_cvat.py` to convert between annotation formats and neatly zip up your dataset to easily upload to CVAT.

## Directory Structure

```
.
├── datasets/
│   ├── my_dataset/
│   │   ├── images/
│   │   │   ├── img001.jpg
│   │   │   └── ...
│   │   ├── labels/
│   │   │   ├── img001.txt
│   │   │   └── ...
│   │   ├── all_images.txt
│   │   ├── data.yaml
│   │   └── autosplit_train.txt
│   └── ...
├── runs/
│   └── my_project/
│       ├── my_model/
│       │   └── weights/
│       │       ├── best.pt
│       │       └── last.pt
│       └── ...
└── videos/
    ├── video1.mp4
    ├── video2.mp4
    └── output/
        ├── video1_out.mp4
        └── ...
```

## Setup

### Dependencies

Make sure you've installed [`Pytorch`](https://pytorch.org/get-started/) according to your system requirements.

**Install ultralytics:**

```sh
pip install ultralytics
```

## Example Commands

- **Extract frames from a video:**
  ```sh
  python extract_frames.py videos/video1.mp4 datasets/new_dataset/images
  ```

- **Generate labels:**
  ```sh
  python generate_labels.py datasets/new_dataset runs/my_project/my_model/weights/best.pt --save --autosplit --split 0.8 0.2 0
  ```

- **Start model training queue:**
  ```sh
  python training.py
  ```

- **Batch predict videos:**
  ```sh
  python batch_predict_video.py videos/ videos/output/ runs/my_project/
  ```

## Notes

- All scripts accept `--help` for usage instructions.
- Place your datasets in the `datasets/` directory and trained models in `runs/`.