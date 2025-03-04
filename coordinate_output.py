import os

from ultralytics import YOLO
import cv2

import time

videoCapture = cv2.VideoCapture(0)
ret, frame = videoCapture.read()

model_path = os.path.join('.', 'runs', 'detect', 'train24', 'weights', 'last.pt')

# Load a model
model = YOLO(model_path)  # load a custom model

threshold = 0.5

prev_frame_time = 0
curr_frame_time = 0

while ret:

    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        midpoint_x = (x1+x2)/2
        midpoint_y = (y1+y2)/2

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 4)
            
            # Horizontal line
            cv2.line(frame,(int(x1),int(midpoint_y)),(int(x2),int(midpoint_y)),(255,0,0),2)
            
            # Vertical line
            cv2.line(frame,(int(midpoint_x),int(y1)),(int(midpoint_x),int(y2)),(255,0,0),2)
            
            cv2.putText(frame, results.names[int(class_id)].upper()+" "+str(int(midpoint_x))+" "+str(int(midpoint_y)), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    # Calculating FPS
    curr_frame_time = time.time()
    fps = 1/(curr_frame_time - (prev_frame_time))
    prev_frame_time = curr_frame_time
    
    # Display FPS
    cv2.putText(frame, str(int(fps)), (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)

    cv2.imshow("Live Basket Detection", frame)

    # Press q to quit the application
    if cv2.pollKey() & 0xFF == ord('q'):
        break

    # Read next frame
    ret, frame = videoCapture.read()

videoCapture.release()
cv2.destroyAllWindows()
