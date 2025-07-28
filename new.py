import cv2

cap = cv2.VideoCapture('/dev/video0')
if not cap.isOpened():
    print("Failed to open /dev/video0")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break
    cv2.imshow("Camera Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
