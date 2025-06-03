from ultralytics import YOLO, settings
import os
from pathlib import Path
from time import sleep
import shutil


if __name__ == '__main__':
    os.environ["MLFLOW_TRACKING_URI"] = 'http://192.168.193.220:5000'

    settings.update({"mlflow": True})

    train_queue_dir = Path('./train-queue')
    
    while (1):
        train_queue = list(train_queue_dir.glob('*.yaml'))

        if train_queue:
            print('Training config found!')

            for config in train_queue:
                print(f'Working on {config}')

                model = YOLO("yolo11n.yaml")    # Load a COCO-pretrained YOLO11n model

                results = model.train(cfg=config)

                shutil.move(config, train_queue_dir / 'done')
        else:
            print('No config found. Waiting 60 secs...')
            sleep(60)