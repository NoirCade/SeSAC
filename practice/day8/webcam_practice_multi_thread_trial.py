import cv2
import dlib
import math
import threading
from queue import Queue
from gtts import gTTS
from playsound import playsound

def process_face(dets, image, output_queue):
    predictor = dlib.shape_predictor(r'C:\Users\bluecom015\Desktop\SeSAC\data\shape_predictor_68_face_landmarks.dat')

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

        # 결과를 큐에 넣어서 반환
        output_queue.put((image.copy(), radian))

def cam_cap():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Could not open webcam")
        exit()

    output_queue = Queue()

    def thread_function():
        while True:
            ret, frame = cap.read()

            if ret:
                frame = cv2.flip(frame, 1)

                detector = dlib.get_frontal_face_detector()
                dets = detector(frame, 1)

                thread_list = []

                for _ in range(len(dets)):
                    thread = threading.Thread(target=process_face, args=(dets, frame, output_queue))
                    thread_list.append(thread)
                    thread.start()

                for thread in thread_list:
                    thread.join()

                # 결과 수집 및 표시 (적절한 방식으로 결과를 표시해야 함)
                while not output_queue.empty():
                    result_frame, result_radian = output_queue.get()
                    cv2.imshow("test", result_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    thread = threading.Thread(target=thread_function)
    thread.start()

    thread.join()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    cam_cap()
