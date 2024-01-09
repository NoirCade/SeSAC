import cv2
import mediapipe as mp
import numpy as np
import sys
import pyautogui
import time

# 전역 변수 초기화
is_index_bent = False  # 검지 손가락이 구부러진 상태 여부를 나타내는 변수
is_click_performed = False  # 클릭 동작을 수행했는지 여부를 나타내는 변수
hand_x, hand_y = 1174, -545  # 손 좌표 초기화
screen_width, scren_height = pyautogui.size()

def initialize_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Webcam not found or not opened.")
        sys.exit()



    return cap

def process_hand(frame, hands):
    global is_index_bent, is_click_performed, hand_x, hand_y, results

    # 손 감지
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # 화면에 손가락 감지 결과 표시
    if results.multi_hand_landmarks:
        # print(results.multi_hand_landmarks)
        pyautogui.FAILSAFE = False  # fail-safe 비활성화
        for hand_landmarks in results.multi_hand_landmarks:
            # 검지 손가락 추적
            index_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_MCP]
            index_tip_2 = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]

            # 검지 손가락의 좌표를 이용하여 커서 위치 업데이트
            hand_x, hand_y = int(index_tip.x * frame.shape[1]*1.05), int(index_tip.y * frame.shape[0]*1.05)

            # 검지 손가락의 y 좌표를 이용하여 구부러진 상태 여부 확인
            is_index_bent = index_tip_2.y < hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_DIP].y

            # 검지 손가락 구부러짐을 기준으로 클릭 동작 감지 및 수행
        if is_index_bent and not is_click_performed:
            # 클릭 동작 수행 (원하는 로직 추가)
            print("Left click performed!")
            is_click_performed = True

            # 손 좌표 업데이트
            pyautogui.click()
            hand_x, hand_y = int(index_tip.x * frame.shape[1]*1.05), int(index_tip.y * frame.shape[0]*1.05)
            print(hand_x, hand_y)
            
        elif not is_index_bent:
            # 클릭 동작 초기화
            is_click_performed = False

        cv2.circle(frame, (hand_x, hand_y), 10, (0, 255, 0), -1)




#-------------------------------------------------------------------------------

# 격자무늬와 원 그리기 함수
def draw_grid_and_circles(img, rows, cols, clicked_points=None, current_point=None, transparency=0.5):
    global cell_size_x, cell_size_y
    height, width, _ = img.shape
    cell_size_x = width // cols
    cell_size_y = height // rows

    # 수직선 그리기
    #for i in range(1, cols):
    #    x = i * cell_size_x
    #    cv2.line(img, (x, 0), (x, height), (255, 255, 255), 1)

    # 수평선 그리기
    #for i in range(1, rows):
    #    y = i * cell_size_y
    #    cv2.line(img, (0, y), (width, y), (255, 255, 255), 1)

    # 마우스 클릭된 지점에서 가장 가까운 교차점 찾기
    if clicked_points is not None:
        for point in clicked_points[0]:
            closest_x = point[0] * cell_size_x
            closest_y = point[1]* cell_size_y
            cv2.circle(img, (closest_x, closest_y), 12, (0, 0, 0), -1)

        for point in clicked_points[1]:
            closest_x = point[0] * cell_size_x
            closest_y = point[1]* cell_size_y
            cv2.circle(img, (closest_x, closest_y), 12, (255, 255, 255), -1)

    # 현재 마우스 위치에 반투명 원 그리기 (미리 보이기)
    if current_point is not None:
        ccx = round(current_point[0] / cell_size_x) * cell_size_x
        ccy = round(current_point[1] / cell_size_y) * cell_size_y
        overlay = img.copy()
        cv2.circle(overlay, (ccx, ccy), 15, (255, 255, 255, 10), -1)
        cv2.addWeighted(overlay, transparency, img, 1 - transparency, 0, img)


# 마우스 클릭 이벤트 및 이동 이벤트 콜백 등록
def mouse_event(event, x, y, flags, param):
    global clicked_points, current_point, cur_player
    winner={0:'Black', 1:'White'}
    color={0:(0,0,0), 1:(255,255,255)}
    chk=0


    if event == cv2.EVENT_LBUTTONDOWN:
        if is_valid_click(x, y, clicked_points):
            clicked_points[cur_player].append([round(x/cell_size_x), round(y/cell_size_y)])
            print(clicked_points)
        
        if check_winner(clicked_points[cur_player]):
            chk=1
        else:
            cur_player = 1 - cur_player

        if chk:
            time.sleep(3)
            reset_game()

    if event == cv2.EVENT_MOUSEMOVE:
        current_point = [x, y]

def is_valid_click(x, y, clicked_points):
    if round(x/cell_size_x)==0 or round(y/cell_size_y)==0 \
                or round(x/cell_size_x)==20 or round(y/cell_size_y)==20:
        return False
    
    for player_points in clicked_points:
        for point in player_points:
            if (abs(point[0] - x) < 15 and abs(point[1] - y) < 15):
                return False
    return True

# 승리조건 확인 함수
def check_winner(points):
    if len(points) < 5:
        return False

    sorted_points = sorted(points)

    for i in range(len(sorted_points) - 4):
        for j in range(i + 1, len(sorted_points) - 3):
            for k in range(j + 1, len(sorted_points) - 2):
                for l in range(k + 1, len(sorted_points) - 1):
                    for m in range(l + 1, len(sorted_points)):
                        if is_winning_combination(sorted_points[i], sorted_points[j], sorted_points[k], sorted_points[l], sorted_points[m]):
                            print('게임 종료')
                            return True
    return False

def is_straight_line(p1, p2, p3, p4, p5):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    x5, y5 = p5

    # 두 점 간의 기울기 계산
    slope1 = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else float('inf')
    slope2 = (y3 - y2) / (x3 - x2) if (x3 - x2) != 0 else float('inf')
    slope3 = (y4 - y3) / (x4 - x3) if (x4 - x3) != 0 else float('inf')
    slope4 = (y5 - y4) / (x5 - x4) if (x5 - x4) != 0 else float('inf')

    # 모든 기울기가 같거나 수직이면 True 반환
    return slope1 == slope2 == slope3 == slope4 or \
           (slope1 == slope2 == slope3 == float('inf') and x1 == x2 == x3 == x4 == x5)

def is_continuous_line(p1, p2, p3, p4, p5):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    x5, y5 = p5

    # 두 점 간의 거리 계산
    distance1 = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    distance2 = np.sqrt((x3 - x2)**2 + (y3 - y2)**2)
    distance3 = np.sqrt((x4 - x3)**2 + (y4 - y3)**2)
    distance4 = np.sqrt((x5 - x4)**2 + (y5 - y4)**2)

    # 거리가 루트2 이하인 경우만 연속된 돌로 판정
    if (distance1 == distance2 == distance3 == distance4) and distance1<=np.sqrt(2):
        return True
    return False

def is_winning_combination(p1, p2, p3, p4, p5):
    if is_straight_line(p1, p2, p3, p4, p5) and is_continuous_line(p1, p2, p3, p4, p5):
        return True
    return False

def reset_game():
    global clicked_points, current_point, cur_player, chk
    clicked_points = [[], []]
    current_point = None
    cur_player = 0
    chk=0



#----------------------------------------------------------------------------


def overlay_images(background, overlay, x, y):
    # 오버레이 영역 계산
    h, w = overlay.shape[:2]
    y_end, x_end = y + h, x + w

    # 작은 이미지를 큰 이미지에 맞춰 크기 조정
    overlay_resized = cv2.resize(overlay, (w, h))

    # 오버레이를 배경 이미지에 더함
    for c in range(0, min(background.shape[2], overlay_resized.shape[2])):
        overlay_channel = overlay_resized[:, :, c]
        background[y:y_end, x:x_end, c] = \
            background[y:y_end, x:x_end, c] * (1 - overlay_channel / 255.0) + \
            overlay_channel * (overlay_channel / 255.0)

#_______________________________________________________________________

def main():
    global hand_x, hand_y
    
    cap = initialize_webcam()

    

    ##창 생성
    cv2.namedWindow('Play')
    cv2.namedWindow('Omok')
    
    ## 초기화
    reset_game()

    ## 마우스 콜백 이벤트 설정
    cv2.setMouseCallback('Play', mouse_event)

    with mp.solutions.hands.Hands() as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            if not ret:
                continue
            
            process_hand(frame, hands)

            # 현재 마우스 위치 확인
            current_x, current_y = pyautogui.position()

            # 마우스 위치가 변경되었을 때만 이동
            if current_x != hand_x or current_y != hand_y:
                pyautogui.moveTo(hand_x, hand_y)

            blended = cv2.imread('Blank_Go_board.png')
            blended = cv2.resize(blended, (640,480))

            background = np.zeros((480, 640, 3), dtype=np.uint8)
            x = (blended.shape[1] - background.shape[1]) // 2
            y = (blended.shape[0] - background.shape[0]) // 2
            
            
            overlay_images(background ,blended,  x, y)


            ## 격자무늬와 원 그리기
            draw_grid_and_circles(frame, 20, 20, clicked_points, current_point, transparency=0.5)
            draw_grid_and_circles(background, 20, 20, clicked_points, current_point, transparency=0.5)
            
            #오목화면 만들기
            background_resized = cv2.resize(background, (background.shape[1], background.shape[0]))
            
            ## 화면에 표시
            cv2.imshow('Omok', background_resized)
            cv2.imshow('Play', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
