import cv2
import time

# Replace with your RTSP stream URL
rtsp_url = ""

# Open stream
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Failed to open RTSP stream.")
else:
    # --- Capture 10 frames, 1 per second ---
    for i in range(10):
        ret, frame = cap.read()
        if ret:
            filename = f"frame_{i}.jpg"
            cv2.imwrite(filename, frame)
            print(type(frame))
            print(f"Saved {filename}")
        else:
            print(f"Failed to read frame {i}")
        time.sleep(1)

    # --- Setup video writer for last 10 seconds ---

    # Output video file
    video_output = 'last_10s.mp4'
    fps = 25  # assumed frame rate
    duration_seconds = 10
    frame_count = fps * duration_seconds

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
    out = cv2.VideoWriter(video_output, fourcc, fps, (width, height))

    print(f"Recording last {duration_seconds} seconds...")

    # --- Record 10 seconds of video ---
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            print(f"Frame drop at {i}")
            continue
        out.write(frame)

    print(f"Video saved to {video_output}")

    # --- Cleanup ---
    out.release()
    cap.release()
