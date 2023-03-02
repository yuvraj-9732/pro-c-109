import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Flip the frame horizontally for a more natural selfie-view
    frame = cv2.flip(frame, 1)

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect the hand using a cascade classifier
    cascade = cv2.CascadeClassifier('palm.xml')
    hands = cascade.detectMultiScale(gray, 1.3, 5)

    # If a hand is detected, take a screenshot and save it to disk
    if len(hands) > 0:
        cv2.imwrite('screenshot.png', frame)
        print("Screenshot taken!")
        break

    # Display the video feed with the detected hand highlighted
    for (x, y, w, h) in hands:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Hand detection', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
