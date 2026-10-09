import cv2

url = input("Enter your phone video URL: ").strip()
camera = cv2.VideoCapture(url)

if not camera.isOpened():
    print("Error: Could not connect to phone camera.")
else:
    print("Camera connected! Press Q to close.")

    while True:
        success, frame = camera.read()

        if not success:
            print("Unable to read video.")
            break

        cv2.imshow("Phone Camera Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()
