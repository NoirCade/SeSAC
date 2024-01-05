import cv2
import time
import dlib
import math
from gtts import gTTS
from playsound import playsound

def cam_cap():
    cap=cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # fps=cap.get(cv2.CAP_PROP_FPS)
    # delay=int(1000/fps)

    if not cap.isOpened():
        print("Could not open webcam")
        exit()

    while True:
        ret, frame = cap.read()

        if ret:
            frame=cv2.flip(frame, 1)
            # img=haar_face(frame)
            img, rad = dlib_face(frame)
            cv2.imshow("test", img)
            if abs(rad)>15:
                sound_out()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def haar_face(img):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    gray_face=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray_face, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)

        roi_color=img[y:y+h, x:x+w]
        roi_gray=gray_face[y:y+h, x:x+w]
        eyes=eye_cascade.detectMultiScale(roi_gray, minSize=(50,30))
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)
    
    return img


def dlib_face(img):
    predictor = dlib.shape_predictor(r'C:\Users\bluecom015\Desktop\SeSAC\data\shape_predictor_68_face_landmarks.dat')
    detector = dlib.get_frontal_face_detector()

    image = img
    dets = detector(image, 1)
    radian = 0.0

    for k,d in enumerate(dets):
        shape = predictor(image, d)
        
        # 얼굴 영역 표시
        color_f=(0,0,255)   # 얼굴
        color_l_out=(255,0,0)   # 랜드마크 바깥쪽
        color_l_in=(0,255,0)    # 랜드마크 안쪽

        # 영역 표시할 선과 원
        line_width=3
        circle_r=3

        # 표시 폰트
        fontType=cv2.FONT_HERSHEY_SIMPLEX
        fontSize=2

        # 얼굴에 사각형 표시
        cv2.rectangle(image, (d.left(), d.top()), (d.right(), d.bottom()), color_f, line_width)

        # 랜드마크에 점 표시
        num_of_points_out=17
        num_of_points_in=shape.num_parts-num_of_points_out
        gx_out=0; gy_out=0; gx_in=0; gy_in=0;

        # shape.part(번호)에 들어있는 좌표에 점 하나씩 찍는다
        for i in range(shape.num_parts):
            shape_point=shape.part(i)
            
            if i<num_of_points_out:
                # cv2.circle(image, (shape_point.x, shape_point.y), circle_r, color_l_out, line_width)
                gx_out=gx_out+shape_point.x/num_of_points_out
                gy_out=gy_out+shape_point.y/num_of_points_out

            else:
                # cv2.circle(image, (shape_point.x, shape_point.y), circle_r, color_l_in, line_width)
                gx_in=gx_in+shape_point.x/num_of_points_in
                gy_in=gy_in+shape_point.y/num_of_points_in
            
        # 랜드마크 점들 중 중심점 표시
        # cv2.circle(image, (int(gx_out), int(gy_out)), circle_r, (0,0,255), line_width)
        # cv2.circle(image, (int(gx_in), int(gy_in)), circle_r, (0,0,0), line_width)

        # 얼굴 방향 계산
        theta=math.asin(2*(gx_in-gx_out)/(d.right()-d.left()))
        radian=theta*180/math.pi

        # print('얼굴 방향: {0:.3f}, 각도: {1:.3f}도'.format(theta, radian))
        
        # if radian<0:
        #     textPrefix='left '
        # else:
        #     textPrefix='right '

        # textShow=textPrefix+str(round(abs(radian), 1))+' deg.'
        # cv2.putText(image, textShow, (d.left(), d.top()), fontType, fontSize, color_f, line_width)

    return image, radian

def sound_out():
    f = open(r'C:\Users\bluecom015\Desktop\SeSAC\practice\day8\face_front.txt', 'r')
    text = f.readline()
    f.close()

    tts = gTTS(text=text, lang='ko')
    tts.save('./tts.mp3')

    playsound('./tts.mp3')

if __name__ == '__main__':
    cam_cap()