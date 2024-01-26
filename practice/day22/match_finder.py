from glob import glob
import os
import shutil

def get_name_list(path):
    file_path = path
    file_list = glob(file_path + '/*.*')
    file_name_list = []

    for file in file_list:
        # file_name = file.split('.')[0]+file.split('.')[1]
        file_name = file.rsplit('_',2)[0]
        file_name_list.append(file_name)

    return file_name_list


def find_common_files(img_folder, json_folder, destination_folder):

    # # 이미지 폴더의 파일 목록 얻기
    # img_files = set(get_name_list(img_folder))

    # # json 폴더의 파일 목록 얻기
    # json_files = set(get_name_list(json_folder))

    # 양쪽에 모두 존재하는 파일명 찾기
    # common_files = img_files.intersection(json_files)
    common_files = list(filter(lambda x: x in get_name_list(json_folder), get_name_list(img_folder)))


    # 저장할 폴더가 없으면 생성
    # if not os.path.exists(destination_folder):
    #     os.makedirs(destination_folder)

    print(list(img_files)[0])
    print(list(json_files)[0])
    print(len(common_files))
    return 0

    # 공통 파일을 대상 폴더로 복사
    for file_name in common_files:
        img_path = os.path.join(img_folder, file_name)
        destination_path = os.path.join(destination_folder, file_name)
        shutil.copy2(img_path, destination_path)
        # shutil.copy2는 파일 속성 및 타임스탬프도 복사합니다.

    print("공통 파일이 성공적으로 복사되었습니다.")


if __name__ == "__main__":

    # 실제 경로 입력
    img_folder = r"C:\Users\bluecom015\Downloads\soccer_data\data\Training\source\selected"
    json_folder = r"C:\Users\bluecom015\Downloads\soccer_data\data\Training\label\labels"
    destination_folder = r"C:\Users\bluecom015\Downloads\soccer_data\data\Training\source\final"

    find_common_files(img_folder, json_folder, destination_folder)