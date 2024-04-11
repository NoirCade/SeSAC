import cv2
import os

if __name__ == '__main__':
    images = []
    for filename in os.listdir('./data'):
        if filename.endswith('.jpg'):
            img_path = os.path.join('./data', filename)
            img = cv2.imread(img_path)
            images.append(img)

    if len(images) > 0:
        result = cv2.hconcat(images)
        cv2.imwrite('merged_image.jpg', result)
        print("Images merged successfully!")
    else:
        print("No images found in the directory.")
