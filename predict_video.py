import os

from ultralytics import YOLO
import cv2


VIDEOS_DIR = os.path.join('.', 'videos')
video_name = 'basket2'
train_name = 'yolo11-nano-hoop'
video_path = os.path.join(VIDEOS_DIR, video_name + '.mp4')
video_path_out = os.path.join(VIDEOS_DIR, 'output', '{}_{}_out.mp4'.format(video_name, train_name))

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
H, W, _ = frame.shape
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

model_path = os.path.join('.', 'runs', 'detect', train_name, 'weights', 'best.pt')

total_inference_time = 0.0
frames = 0

# Load a model
model = YOLO(model_path)  # load a custom model

threshold = 0.5

while ret:

    results = model(frame)[0]
    total_inference_time += results.speed['preprocess']
    total_inference_time += results.speed['inference']
    total_inference_time += results.speed['postprocess']
    frames += 1

    # Get frame time
    frame_time = 0.0
    frame_time += results.speed['preprocess']
    frame_time += results.speed['inference']
    frame_time += results.speed['postprocess']

    # Calculate FPS
    fps = 1000/frame_time

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 3, cv2.LINE_AA)
    
    # Display FPS
    cv2.putText(frame, str(int(fps)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    out.write(frame)
    ret, frame = cap.read()

print()
print("Average Inference Time per frame: ", (total_inference_time/frames))
print("FPS based on Average Inference Time: ", (1/(total_inference_time/(frames*1000))))

cap.release()
out.release()
cv2.destroyAllWindows()
