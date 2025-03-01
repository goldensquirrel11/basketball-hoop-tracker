from ultralytics import YOLO

if __name__ == '__main__':

    model = YOLO("yolo11n.yaml")

    results = model.train(data="config.yaml", epochs=10000)