import cv2
import os

if __name__ == '__main__':
    img = cv2.imread('result.jpg')
    y = int(img.shape[0]/3)
    x = int(img.shape[1]/5)
    count = 1

    os.makedirs('./data', exist_ok=True)
    for y_count in range(3):
        for x_count in range(5):
            print(y, x)
            split = img[y*y_count:y*(y_count+1)][x*x_count:x*(x_count+1)]
            cv2.imwrite(f'./data/imgae{count}.jpg', split)