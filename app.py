import cv2
import torch
import yt_dlp

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Access YouTube video feed using yt-dlp
url = "https://www.youtube.com/watch?v=HtNji7rdV0E"
ydl_opts = {'format': 'best'}
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info_dict = ydl.extract_info(url, download=False)
    best_stream_url = info_dict['url']  # Direct video stream URL

capture = cv2.VideoCapture(best_stream_url)

while True:
    ret, frame = capture.read()
    if not ret:
        break

    # Perform object detection
    results = model(frame)

    # Extract results
    labels, cord = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()

    # Plot bounding boxes and labels on the frame
    for i in range(len(labels)):
        row = cord[i]
        if row[4] < 0.2:  # Adjust confidence threshold as needed
            continue
        x1, y1, x2, y2 = int(row[0] * frame.shape[1]), int(row[1] * frame.shape[0]), int(row[2] * frame.shape[1]), int(row[3] * frame.shape[0])
        bgr = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
        cv2.putText(frame, model.names[int(labels[i])], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

    cv2.imshow("Object Detection", frame)

    if cv2.waitKey(1) == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
