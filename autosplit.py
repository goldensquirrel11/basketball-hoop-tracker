from ultralytics.data.utils import autosplit
autosplit(
    path="datasets/images/train/",
    weights=(0.8, 0.1, 0.1)
)