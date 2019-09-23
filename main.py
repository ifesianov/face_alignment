import os
import dlib
from File import File
from LandmarksDetector import LandmarksDetector
from Alignment import Alignment

IMG_PATH = 'images/data'
LAND_PATH = 'landmarks'

DEFORMED_IMG_PATH = 'deformed_images'


class ApplyMultipleAlignment:
    LANDMARKS_PREDICTION_MODEL = os.path.join('landmarks_model', 'shape_predictor_68_face_landmarks.dat')
    IDEAL_MASK_FILE_NAME = 'average_face_points.txt'

    def __init__(self, land_path, img_path, deformed_img_path):
        self.land_path = land_path
        self.img_path = img_path
        self.deformed_img_path = deformed_img_path

        self._ideal_mask = None
        self._face_detector = None
        self._face_pose_predictor = None

        self._img_dir_names = None

    @property
    def ideal_mask(self):
        if self._ideal_mask is None:
            self._ideal_mask = self.read_ideal_mask()
        return self._ideal_mask

    @property
    def img_dir_names(self):
        if self._img_dir_names is None:
            self._img_dir_names = File.get_img_dir_name(self.img_path)
        return self._img_dir_names

    def read_ideal_mask(self):
        return File.read_land('', self.IDEAL_MASK_FILE_NAME)

    @property
    def face_detector(self):
        if self._face_detector is None:
            self._face_detector = dlib.get_frontal_face_detector()
        return self._face_detector

    @property
    def face_pose_predictor(self):
        if self._face_pose_predictor is None:
            self._face_pose_predictor = dlib.shape_predictor(self.LANDMARKS_PREDICTION_MODEL)
        return self._face_pose_predictor

    def run(self):
        number_of_dirs = len(self.img_dir_names)
        count = 0
        for dir_name in self.img_dir_names:
            img_file_names = File.get_img_file_names(os.path.join(self.img_path, dir_name))

            for file_name in img_file_names:
                LandmarksDetector(land_path=LAND_PATH,
                               img_path=IMG_PATH,
                               face_pose_predictor=self.face_pose_predictor,
                               face_detector=self.face_detector,
                               file_name=file_name,
                               ).run()
                Alignment(land_path=LAND_PATH,
                       img_path=IMG_PATH,
                       deformed_img_path=DEFORMED_IMG_PATH,
                       ideal_mask=self.ideal_mask,
                       file_name=file_name
                       ).run()
            if count % 10 == 0:
                print('{}/{}'.format(count, number_of_dirs))
            print()


if __name__ == '__main__':
    print('Init all')
    obj = ApplyMultipleAlignment(land_path=LAND_PATH,
                                 img_path=IMG_PATH,
                                 deformed_img_path=DEFORMED_IMG_PATH)
    print('Start')
    obj.run()
    print('Done!')
