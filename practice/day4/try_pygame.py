import pygame
import sys

# Pygame 초기화
pygame.init()

# 화면 설정
screen_width, screen_height = 800, 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("공 받아치기")

# 색깔 정의
white = (255, 255, 255)
black = (0, 0, 0)

# 공 설정
ball_radius = 15
ball_x, ball_y = screen_width // 5, screen_height // 5
ball_speed_x, ball_speed_y = 7, 7

# 패들 설정
paddle_width, paddle_height = 100, 10
paddle_x, paddle_y = (screen_width - paddle_width) // 2, screen_height - 2 * paddle_height
paddle_speed = 8

# 점수 초기화
score = 0

# 폰트 설정
font = pygame.font.Font(None, 36)

# 게임 루프
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT] and paddle_x - paddle_speed > 0:
        paddle_x -= paddle_speed
    if keys[pygame.K_RIGHT] and paddle_x + paddle_speed < screen_width - paddle_width:
        paddle_x += paddle_speed

    # 공 이동
    ball_x += ball_speed_x
    ball_y += ball_speed_y

    # 벽과의 충돌
    if ball_x - ball_radius <= 0 or ball_x + ball_radius >= screen_width:
        ball_speed_x = -ball_speed_x
    if ball_y - ball_radius <= 0:
        ball_speed_y = -ball_speed_y

    # 패들과의 충돌
    if (
        paddle_x <= ball_x <= paddle_x + paddle_width
        and paddle_y <= ball_y + ball_radius <= paddle_y + paddle_height
    ):
        ball_speed_y = -ball_speed_y
        score += 1

    # 공이 화면 아래로 내려갔을 때 재시작
    if ball_y - ball_radius > screen_height:
        ball_x, ball_y = screen_width // 5, screen_height // 5
        score = 0

    # 화면 초기화
    screen.fill(black)

    # 공과 패들 그리기
    pygame.draw.circle(screen, white, (ball_x, ball_y), ball_radius)
    pygame.draw.rect(screen, white, (paddle_x, paddle_y, paddle_width, paddle_height))

    # 점수 표시
    score_text = font.render(f"Score: {score}", True, white)
    screen.blit(score_text, (10, 10))

    # 화면 업데이트
    pygame.display.flip()

    # 초당 프레임 설정
    pygame.time.Clock().tick(60)
