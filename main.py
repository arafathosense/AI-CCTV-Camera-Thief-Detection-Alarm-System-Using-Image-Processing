import cv2
import winsound
import time

# File paths
video_path = r"C:\\Users\\iTparK\\Desktop\\New folder\\Home Security Camera\\thiefs.mp4"
alarm_sound_path = r'C:\\Users\\iTparK\\Desktop\\New folder\\Home Security Camera\\alert.wav'

# Open video
cam = cv2.VideoCapture(video_path)

if not cam.isOpened():
    print("Error: Could not open video file.")
    exit()

# Read the first frame for reference
ret, frame_prev = cam.read()
frame_prev = cv2.GaussianBlur(cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY), (5, 5), 0)

alarm_on = False  # To prevent continuous alarm repeats
last_alarm_time = 0  # Alarm delay timer

while True:
    ret, frame = cam.read()
    if not ret:
        print("Video finished.")
        break

    # Prepare current frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Frame difference
    diff = cv2.absdiff(frame_prev, gray)

    # Threshold
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    # Dilation to fill gaps
    dilated = cv2.dilate(thresh, None, iterations=3)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    motion_detected = False

    for c in contours:
        if cv2.contourArea(c) < 2500:  # Minimum area filter
            continue

        # Bounding box (RED COLOR)
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # ← red box

        motion_detected = True

    # Play alarm (once every 2 sec)
    if motion_detected:
        if not alarm_on or time.time() - last_alarm_time > 2:
            winsound.PlaySound(alarm_sound_path, winsound.SND_ASYNC)
            alarm_on = True
            last_alarm_time = time.time()
    else:
        alarm_on = False

    # Show output
    cv2.imshow("Security Camera", frame)

    # Key controls
    key = cv2.waitKey(20)
    if key == ord('q'):
        break
    elif key == ord('p'):
        cv2.waitKey(0)

    # Update previous frame
    frame_prev = gray.copy()

cam.release()
cv2.destroyAllWindows()
