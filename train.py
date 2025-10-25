from ultralytics import YOLO
import os

if __name__ == '__main__':
    os.environ["MLFLOW_TRACKING_URI"] = 'http://192.168.193.220:5000'

    # model = YOLO("runs/detect/train3/weights/last.pt")  # For resuming a previous training session
                                                        # OR using a previous pretrained model weights
    model = YOLO("yolo11n.yaml")    # Load a COCO-pretrained YOLO11n model

    results = model.train(data="config.yaml",
                        #   name="train3",    # Name of training run
                        #   exist_ok=True,    # Allows overwriting an existing project/directory
                          epochs=500,       # Number of training epochs
                          time=24,            # Maximum training hours (overrides epochs argument)
                          batch=0.99,       # GPU memory utilization %
                        #   cache=True,       # Caches dataset images in RAM (need lots of RAM)
                          single_cls=True,  # Treats datasets as a single class (useful for object presence rather than classification)
                          resume=True,      # Resumes training from the last saved checkpoint
                          profile=True,     # Enables profiling of ONNX and TensorRT speeds during training
                          plots=True        # Generates & saves plots of training & validation metrics as well as prediction examples
                          )