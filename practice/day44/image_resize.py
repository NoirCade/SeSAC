import zipfile
import os
import cv2
import numpy as np

# 압축되어있는 데이터를 풀고, 기존 2800*1024 사이즈의 데이터를 종횡비 유지한 채로 검은색 패딩을 넣어 640*640으로 리사이즈
def ext_resize(comp_file, target_width, target_height):
    with zipfile.ZipFile(comp_file, 'r') as zip_ref:
        zip_ref.extractall('extracted_images')

    # 압축 해제된 이미지들에 대해 조정
    for filename in zip_ref.namelist():
        # 이미지 파일인 경우에만 처리
        if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
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