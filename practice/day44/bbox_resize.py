from glob import glob
import cv2
import numpy as np

def bbox_resize(label_path):
    label_list = glob(label_path)

    for label in label_list:
        boxes = []
        # print(label.split('labels'))
        with open(label, 'r') as f:
            for line in f:
                boxes.append(line.strip())

        # matching_path = label.split('\\')[0] + '/' + label.split('\\')[1] + '/images/' + label.split('\\')[-1].split('.')[0] + '.jpg'
        # matching_img = cv2.imread(matching_path)
        
        # print('old_boxes == ', boxes)
        for idx, box in enumerate(boxes):
            l = box.split(' ')[0]
            x = float(box.split(' ')[1])
            y = float(box.split(' ')[2])
            w = float(box.split(' ')[3])
            h = float(box.split(' ')[4])
            resized_y = ((y * 1024) + 888) / 2800
            resized_h = (h * 1024) / 2800
            new_box = l + ' ' + str(x) + ' ' + str(resized_y) + ' ' + str(w) + ' ' + str(resized_h) + '\n'
            # print('box == ' + box)
            # print('new_box == ' + new_box)
            boxes[idx] = new_box
            # pt1 = (int((x*640)-(0.5*w*640)), int((resized_y*640)-(0.5*resized_h*640)))
            # pt2 = (int((x*640)+(0.5*w*640)), int((resized_y*640)+(0.5*resized_h*640)))
            # print(pt1, pt2)
            # img = cv2.rectangle(matching_img, pt1, pt2, (75, 255, 0), 3)
            # cv2.imshow('test', img)
            # cv2.waitKey(0)
        # print('new_boxes == ', boxes)
        with open(label, 'w') as f:
            f.writelines(boxes)
        
        print(label + ' file box_resized')


if __name__ == '__main__':
    path = 'C:/Users/kth53/Downloads/resized_woods/*/*/*.txt'
    bbox_resize(path)