from sklearn.model_selection import train_test_split
from glob import glob
import os
import shutil

# 리사이즈된 데이터셋을 train / val / test 6 : 2 : 2 로 split 진행
def ttv_split(img_folder_path):
    img_lists = glob(img_folder_path)
    train_imgs, vt_imgs = train_test_split(img_lists, test_size=0.4, random_state=77, shuffle=True)
    val_imgs, test_imgs = train_test_split(vt_imgs, test_size=0.5, random_state=77, shuffle=True)

    folder_path = 'C:/Users/bluecom015/Desktop/SeSAC/practice/day44/resized_woods/'
    os.makedirs(folder_path + '/train/labels', exist_ok=True)
    os.makedirs(folder_path + '/val/labels', exist_ok=True)
    os.makedirs(folder_path + '/test/labels', exist_ok=True)


    for img in train_imgs:
        shutil.move(img, folder_path + '/train/labels')

    for img in val_imgs:
        shutil.move(img, folder_path + '/val/labels')
    
    for img in test_imgs:
        shutil.move(img, folder_path + '/test/labels')


if __name__ == '__main__':
    ttv_split('./resized_woods/*/*.txt')