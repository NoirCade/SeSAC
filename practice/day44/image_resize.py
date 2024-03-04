import zipfile
import os
import cv2
import numpy as np

def ext_resize(comp_file, target_width, target_height):
    # 압축 파일 열기
    with zipfile.ZipFile(comp_file, 'r') as zip_ref:
        # 압축 해제
        zip_ref.extractall('extracted_images')

    # 압축 해제된 이미지들에 대해 조정
    for filename in zip_ref.namelist():
        # 이미지 파일인 경우에만 처리
        if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
            # 이미지 열기
            image = cv2.imread(f'extracted_images/{filename}')
            height, width = image.shape[:2]
            if width / height > target_width / target_height:
                new_width = target_width
                new_height = int(height * (target_width / width))
            else:
                new_width = int(width * (target_height / height))
                new_height = target_height
            
            # 원하는 크기로 이미지 조정
            resized_image = cv2.resize(image, (new_width, new_height))
            padded_image = np.zeros((target_height, target_width, 3), dtype=np.uint8)

            start_x = (target_width - new_width) // 2
            start_y = (target_height - new_height) // 2
            padded_image[start_y:start_y+new_height, start_x:start_x+new_width] = resized_image
            
            # 조정된 이미지 저장
            os.makedirs('./resized_images/Images - 1/Images - 1', exist_ok=True)
            cv2.imwrite(f'resized_images/{filename}', padded_image)
            print(f'saved resized_images/{filename}')

if __name__ == '__main__':
    ext_resize('C:/Users/bluecom015/Downloads/wood.zip', 640, 640)