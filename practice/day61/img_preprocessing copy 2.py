import cv2
import numpy as np

if __name__ == '__main__':
    img = cv2.imread('coupon.jpg')
    img = cv2.rotate(img, 1)
    
    const = cv2.convertScaleAbs(img, 1.5, 1.2)
    # cv2.imshow('test', const)
    # cv2.imshow('test2', img)
    # cv2.waitKey(0)
    cv2.imwrite('result.jpg', const)