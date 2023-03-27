# Importing the OpenCV library
import cv2

# Initializing the face cascade classifier by providing its file path
face_cascade = cv2.CascadeClassifier('/Users/andreivince/Desktop/Code Folder/Face_Detection_Python/haarcascade_frontalface_default.txt')

# Starting the video capture from the default camera (usually the built-in webcam)
cam = cv2.VideoCapture(0)

# Continuously capturing frames from the camera until the user interrupts
while True:
    # Reading the frame from the camera and returning a boolean value indicating whether the frame was successfully read or not
    ret, img = cam.read()
    
    # Converting the color image to grayscale for efficient face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detecting faces in the grayscale image using the face cascade classifier
    # The scaleFactor and minNeighbors parameters control the sensitivity and accuracy of the face detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=4)

    # Drawing rectangles around the detected faces in the original color image
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Displaying the color image with detected faces in a window named "img"
    cv2.imshow("img", img)
    
    # Waiting for a key press for 30 milliseconds and storing the key code in variable k
    k = cv2.waitKey(30) & 0xff
    
    # Checking if the user pressed the 'ESC' key (key code 27) to exit the loop
    if k == 27:
        break

# Releasing the camera resources and destroying all open windows
cam.release()
cv2.destroyAllWindows()

