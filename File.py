import os
import cv2
import numpy as np


class File:
    @staticmethod
    def get_img_file_names(img_path):
        return [file_name for file_name in os.listdir(img_path)\
                               if os.path.splitext(file_name)[1] in ('.jpg', '.png')]

    @staticmethod
    def land_to_txt(file_name, points):
        with open(os.path.join(file_name), mode='w') as land_file:
            for el in points:
                land_file.write('{} {}\n'.format(int(el[0]), int(el[1])))

    @staticmethod
    def read_land(land_path, file_name):
        points = []
        with open(os.path.join(land_path, file_name)) as file:
            for line in file:
                x, y = line.split()
                points.append((int(x), int(y)))
        return points

    @staticmethod
    def read_img(img_path, file_name, convert_to_float=False):
        img = cv2.imread(os.path.join(img_path, file_name))

        # Convert to floating point
        if convert_to_float:
            img = np.float32(img) / 255.0
        return img

    @staticmethod
    def save_img(file_name, img, convert_to_float=False):
        if convert_to_float:
            img = img * 255
        cv2.imwrite(file_name, img)

    @staticmethod
    def show_img(img, file_name=None):
        cv2.imshow(file_name, img)
        cv2.waitKey(0)