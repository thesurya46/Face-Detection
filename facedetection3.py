import cv2

# Load the face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start video capture
cap = cv2.VideoCapture(0)

# Create a named window and set it to fullscreen or normal
cv2.namedWindow('Face Detection', cv2.WINDOW_NORMAL)  # Allows resizing
cv2.setWindowProperty('Face Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)  # Fullscreen mode

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=6)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 5)

    cv2.imshow('Face Detection', frame)

    if cv2.waitKey(40) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()