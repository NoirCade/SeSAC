from PIL import Image
from glob import glob
import shutil


def find_a0456(img_list, dest_path):
    for image in img_list:
        top_folder = img_path.split('\\')[-3]   # 폴더 구조 받아오기
        folder = image.split(top_folder)
        final_destination = dest_path + folder[-1]
        
        if ('a-04' in image) or ('a-05' in image) or ('a-06' in image):
        # if 'a-05' in image:
            try:
                shutil.copy(image, final_destination)
                print(f"Copied: {image}")
            except PermissionError as e:
                print(f"Permission error while copying {image}: {e}")

        

if __name__ == '__main__':
    ## 작업할 이미지 폴더 경로 설정
    img_path = r'C:\Users\bluecom015\Downloads\soccer_data\gray_data\\'   # 이미지 들어있는 최상위 경로 지정
    img_list = glob(img_path + '*\\*\\*.jpg')   # 하위 폴더들의 모든 이미지들 리스트에 넣고
    # print(len(img_list))  # 이미지 수량 체크

    # 리사이징된 이미지 들어갈 최상위 경로 지정 ** 하위 폴더 미리 만들어주세요
    dest_path = r'C:\Users\bluecom015\Downloads\soccer_data\a0456_gray'  # 마지막 \\ 생략해주세요

    # 함수 실행
    find_a0456(img_list, dest_path)


