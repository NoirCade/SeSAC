import cv2
import numpy as np
import time

# 격자무늬와 원 그리기 함수
def draw_grid_and_circles(img, rows, cols, clicked_points=None, current_point=None, transparency=0.5):
    global cell_size_x, cell_size_y
    height, width, _ = img.shape
    cell_size_x = width // cols
    cell_size_y = height // rows

    # 수직선 그리기
    for i in range(1, cols):
        x = i * cell_size_x
        cv2.line(img, (x, 0), (x, height), (255, 255, 255), 1)

    # 수평선 그리기
    for i in range(1, rows):
        y = i * cell_size_y
        cv2.line(img, (0, y), (width, y), (255, 255, 255), 1)

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
            # print((round(x/cell_size_x), round(y/cell_size_y)), color[cur_player], winner[cur_player])
            # cv2.circle(frame, (round(x/cell_size_x), round(y/cell_size_y)), 12, color[cur_player], -1)
            # cv2.putText(frame, f'{winner[cur_player]} WIN!', (300, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
            # draw_grid_and_circles(frame, 20, 20, clicked_points, current_point, transparency=0.5)
            chk=1
        else:
            cur_player = 1 - cur_player

    if event == cv2.EVENT_MOUSEMOVE:
        current_point = [x, y]

    if chk:
        print((round(x/cell_size_x), round(y/cell_size_y)), color[cur_player], winner[cur_player])
        # cv2.circle(frame, (round(x/cell_size_x), round(y/cell_size_y)), 12, color[cur_player], -1)
        draw_grid_and_circles(frame, 20, 20, clicked_points, current_point, transparency=0.5)
        # cv2.putText(frame, 'WIN!', (300, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        
        time.sleep(3)
        reset_game()
        chk=0

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
    global clicked_points, current_point, cur_player
    clicked_points = [[], []]
    current_point = None
    cur_player = 0



if __name__ == '__main__':
    # 웹캠 열기
    cap = cv2.VideoCapture(0)

    # 창 생성
    cv2.namedWindow('Omok')
    cv2.namedWindow('Play')

    # 초기화
    reset_game()

    # 마우스 콜백 이벤트 설정
    cv2.setMouseCallback('Omok', mouse_event)

    # 메인 루프
    while True:
        # 프레임 읽기
        ret, frame = cap.read()        

        # 격자무늬와 원 그리기
        draw_grid_and_circles(frame, 20, 20, clicked_points, current_point, transparency=0.5)  # 20x20 격자를 그립니다.

        # 화면에 표시
        cv2.imshow('Omok', frame)
        cv2.imshow('Play', frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 종료
    cap.release()
    cv2.destroyAllWindows()
