import json
from glob import glob
import shutil

def file_copy(file_list, direction):
    # 라벨 왼쪽과 오른쪽의 폴더를 지정
    r_path = r'C:\Users\bluecom015\Downloads\soccer_data\data\final_training\Right'
    l_path = r'C:\Users\bluecom015\Downloads\soccer_data\data\final_training\Left'

    if direction:   # direction 값에 따라 저장할 경로 설정
        destination_path = l_path
    else:
        destination_path = r_path
         
    for file_path in file_list:
        try:
            shutil.copy(file_path, destination_path)    # 지정한 경로에 복사 시도
            print(f"Copied: {file_path}")
        except PermissionError as e:    # 예외시 에러 출력
            print(f"Permission error while copying {file_path}: {e}")


def direct_file(file_list, img_path):
    direction = 0   # 오른쪽이 0, 왼쪽이 1로 설정

    for file in file_list:
        img_name = file.split('_b')[0].split('\\')[-1]
        img_list = glob(img_path + '\\' + img_name + '*.jpg')

        if len(img_list)>0: # 가져온 json 파일에 해당하는 이미지 있을 경우 아래 실행
            with open(file) as f:
                data = json.load(f) # json 파일 읽기
                last_y = data["labelingInfo"]["ball"]["location"][-1]['y']  # 마지막 y좌표값 받아옴

                direction = 1 if last_y > 0 else 0  # 읽어온 마지막 y좌표값에 따라 방향 설정
                file_copy(img_list, direction)  # 파일 경로와 방향값 지정하여 복사
    

if __name__ == "__main__":
    img_path = r'C:\Users\bluecom015\Downloads\soccer_data\data\final_training\\'
    json_path = r'C:\Users\bluecom015\Downloads\soccer_data\data\final_label\\'
    json_list = glob(json_path + '*.json')

    direct_file(json_list, img_path)
