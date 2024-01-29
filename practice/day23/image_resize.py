from PIL import Image
from glob import glob

'''
현재 폴더구조
final_training
    ㄴ Right
        ㄴ rf
            ㄴ 이미지 파일들
        ㄴ lf
            ㄴ 이미지 파일들
    ㄴ Left
        ㄴ rf
            ㄴ 이미지 파일들
        ㄴ lf
            ㄴ 이미지 파일들

이 코드 실행 후 폴더구조
resized_training
    ㄴ Right
        ㄴ rf
            ㄴ 이미지 파일들
        ㄴ lf
            ㄴ 이미지 파일들
    ㄴ Left
        ㄴ rf
            ㄴ 이미지 파일들
        ㄴ lf
            ㄴ 이미지 파일들

이런 식으로 최상위 폴더 경로만 지정해주면 기존에 정리해둔 폴더구조 그대로 받아와서 정리하게 코드 작성했습니다
하지만 저장할 경로의 하위폴더들도 미리 만들어주세요
'''

def resize_imgs(img_list, dest_path):
    for image in img_list:
        img = Image.open(image)
        top_folder = img_path.split('\\')[-3]   # 폴더 구조 받아오기
        folder = image.split(top_folder)
        final_destination = dest_path + folder[-1]
        # print(final_destination) # 저장할 경로 확인

        resized_img = img.resize((224,224))
        resized_img.save(final_destination)  ## 리사이즈 후 이미지 저장

if __name__ == '__main__':
    ## 작업할 이미지 폴더 경로 설정
    img_path = r'C:\Users\bluecom015\Downloads\soccer_data\data\final_training\\'   # 이미지 들어있는 최상위 경로 지정
    img_list = glob(img_path + '*\\*\\*.jpg')   # 하위 폴더들의 모든 이미지들 리스트에 넣고
    # print(len(img_list))  # 이미지 수량 체크

    # 리사이징된 이미지 들어갈 최상위 경로 지정 ** 하위 폴더 미리 만들어주세요
    dest_path = r'C:\Users\bluecom015\Downloads\soccer_data\data\resized_training'  # 마지막 \\ 생략해주세요

    # 함수 실행
    resize_imgs(img_list, dest_path)


