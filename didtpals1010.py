import cv2

# 각 변수에 훈련된 얼굴인식 인공지능 xml파일을 넣음.
fece_xml = 'haarcascade_frontalface_default.xml'
eyes_xml = 'haarcascade_eye.xml'

# call 변수에 CascadeClassifier 함수를 사용하여 xml 변수에 있는 인공지는을 불러옴.
call_in_face = cv2.CascadeClassifier(fece_xml)
call_in_eyes = cv2.CascadeClassifier(eyes_xml)

# 다운받은 이미지를 imread 함수를 사용하여 불러옴.
cap = cv2.VideoCapture(0)



while True:
    # Read the frame
    _, img = cap.read()

    img = cv2.flip(img, 2)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    faces = call_in_face.detectMultiScale(    
        gray,             
        scaleFactor=1.05,   # 이미지 스케일 설정
        minNeighbors=5,     # 인접한 객체 최소거리 픽셀
    minSize=(150, 150), # 탐지 객체 최소크기 설정
    )
    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # Display
    cv2.imshow('img', img)
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break

cap.release()
 
