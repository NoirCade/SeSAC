import cv2
import dlib
import math
import threading
from gtts import gTTS
from playsound import playsound

def process_frame(frame):
    img, rad = dlib_face(frame)
    cv2.imshow("test", img)
    if abs(rad) > 15:
        sound_out()

def cam_cap():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Could not open webcam")
        exit()

    while True:
        ret, frame = cap.read()

        if ret:
            frame = cv2.flip(frame, 1)

            # 멀티스레딩을 사용하여 이미지 처리
            t1 = threading.Thread(target=process_frame, args=(frame,))
            t1.start()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

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
        
        if radian<0:
            textPrefix='left '
        else:
            textPrefix='right '

        textShow=textPrefix+str(round(abs(radian), 1))+' deg.'
        cv2.putText(image, textShow, (d.left(), d.top()), fontType, fontSize, color_f, line_width)

    return image, radian

def sound_out():
    playsound(r'C:\Users\bluecom015\Desktop\SeSAC\practice\day8\tts.mp3')

if __name__ == '__main__':
    cam_cap()
