import cv2

# 각 변수에 훈련된 얼굴인식 인공지능 xml파일을 넣음.
fece_xml = 'haarcascade_frontalface_default.xml'
eyes_xml = 'haarcascade_eye.xml'

# call 변수에 CascadeClassifier 함수를 사용하여 xml 변수에 있는 인공지능을 불러옴.
call_in_face = cv2.CascadeClassifier(fece_xml)
call_in_eyes = cv2.CascadeClassifier(eyes_xml)

# 다운받은 이미지를 imread 함수를 사용하여 불러옴.
image = cv2.imread('image.jpg')
 
# detectMultiScale 함수를 사용하여 call 변수에 들어있는 인공지능이 이미지에 얼굴을 인식하게 해줌.
face_rc = call_in_face.detectMultiScale(
    image,              # 이미지를 불러옴.
    scaleFactor=1.05,   # 이미지 스케일 설정
    minNeighbors=5,     # 인접한 객체 최소거리 픽셀
    minSize=(150, 150)  # 객체 탐지
    )

eyes_rc = call_in_eyes.detectMultiScale(
    image,
    scaleFactor=1.05,
    minNeighbors=5,
    minSize=(30, 30),
    )

# 인식된 이미지를 rectangle 함수를 사용하여 이미지에 좌표를 설정하고 설정된 좌표에 직사각형에 start지점과 end지점을 설정해주어 인식된 곳에 직사각형 표시.
# 반복문을 사용하여 위 활동을 반복함.
for (x, y, w, h) in face_rc:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2) # cv2.rectangle(이미지, start_point, end_point, 색상, 두께)

for (x, y, w, h) in eyes_rc:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# 완료된 이미지를 표시.
cv2.imshow('image', image)

# 키를 누르기 전까지 꺼지지 않게함.
cv2.waitKey()

