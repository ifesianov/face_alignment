import dlib
from skimage import io
import os
import numpy as np
from File import File
import matplotlib.pyplot as plt
import face_alignment

# You can download the required pre-trained face detection model here:
# http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2


IMG_PATH = 'images/data/13_1'
LAND_PATH = 'landmarks'

FILE_NAME = '1561122879.5871034.jpg'

LANDMARKS_PREDICTION_MODEL = os.path.join('landmarks_model', 'shape_predictor_68_face_landmarks.dat')


class LandmarksDetector:
    def __init__(self, land_path, img_path, face_detector, face_pose_predictor, file_name):
        self.land_path = land_path
        self.img_path = img_path
        self.img_file_name = file_name
        self._land_file_name = None
        self._img = None
        self._landmarks = None

        self.face_detector = face_detector
        self.face_pose_predictor = face_pose_predictor

    @property
    def land_file_name(self):
        if self._land_file_name is None:
            self._land_file_name = os.path.splitext(self.img_file_name)[0] + '.txt'
        return self._land_file_name

    @property
    def img(self):
        if self._img is None:
            self._img = File.read_img(self.img_path, self.img_file_name)
        return self._img

    @property
    def landmarks(self, predict_face=False):
        if self._landmarks is None:
            if self.face_pose_predictor.__class__.__name__ == 'shape_predictor':
                rect = self.get_face_rect() if predict_face\
                    else dlib.rectangle(left=0, top=0, right=self.img.shape[0], bottom=self.img.shape[1])

                pose_landmarks = self.face_pose_predictor(self.img, rect)
                self._landmarks = np.array([[el.x, el.y] for el in list(pose_landmarks.parts())])
            elif self.face_pose_predictor.__class__.__name__ == 'FaceAlignment':
                self._landmarks = self.face_pose_predictor.get_landmarks(self.img)[-1]
        return self._landmarks

    def get_face_rect(self):
        detected_faces = self.face_detector(self.img, 1)
        if len(detected_faces) > 1:
            print('Image has multiple faces')
        return detected_faces[0]

    def land_to_txt(self, points):
        File.land_to_txt(os.path.join(self.land_path, self.land_file_name), points)

    def show_land_on_img(self):
        fig = plt.figure(figsize=plt.figaspect(.5))
        ax = fig.add_subplot(1, 2, 1)
        ax.imshow(self.img)

        ax.scatter(self.landmarks[:, 0],
                   self.landmarks[:, 1])

        ax.axis('off')

        plt.show()

    def run(self, predict_face=False):
        # print('Start')
        points = self.landmarks
        self.land_to_txt(points)
        # print('Landmarks saved!'

if __name__ == '__main__':
    face_detector = dlib.get_frontal_face_detector()
    # face_pose_predictor = dlib.shape_predictor(LANDMARKS_PREDICTION_MODEL)
    face_pose_predictor = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cpu', flip_input=True)
    lm = LandmarksDetector(land_path=LAND_PATH,
                           img_path=IMG_PATH,
                           file_name=FILE_NAME,
                           face_detector=face_detector,
                           face_pose_predictor=face_pose_predictor)
    lm.run(predict_face=False)
    lm.show_land_on_img()
