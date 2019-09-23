import dlib
from skimage import io
import os
from File import File

# You can download the required pre-trained face detection model here:
# http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2


IMG_PATH = 'images'
LAND_PATH = 'landmarks'

FILE_NAME = 'atynkaliuk.jpg'


class LandmarksDetector:
    PREDICTION_MODEL = os.path.join('landmarks_model', 'shape_predictor_68_face_landmarks.dat')

    def __init__(self, land_path, img_path, file_name):
        self.land_path = land_path
        self.img_path = img_path
        self.img_file_name = file_name
        self._land_file_name = None
        self._img = None

        self._face_detector = None
        self._face_pose_predictor = None

    @property
    def land_file_name(self):
        if self._land_file_name is None:
            self._land_file_name = os.path.splitext(self.img_file_name)[0] + '.txt'
        return self._land_file_name

    @property
    def face_detector(self):
        if self._face_detector is None:
            self._face_detector = dlib.get_frontal_face_detector()
        return self._face_detector

    @property
    def face_pose_predictor(self):
        if self._face_pose_predictor is None:
            self._face_pose_predictor = dlib.shape_predictor(self.PREDICTION_MODEL)
        return self._face_pose_predictor

    @property
    def img(self):
        if self._img is None:
            self._img = File.read_img(self.img_path, self.img_file_name)
        return self._img

    def get_face_rect(self):
        detected_faces = self.face_detector(self.img, 1)
        if len(detected_faces) > 1:
            print('Image has multiple faces')
        return detected_faces[0]

    def land_to_txt(self, points):
        File.land_to_txt(os.path.join(self.land_path, self.land_file_name), points)

    def predict_landmarks(self, predict_face=False):
        rect = self.get_face_rect() if predict_face\
            else dlib.rectangle(left=0, top=0, right=self.img.shape[0], bottom=self.img.shape[1])

        pose_landmarks = self.face_pose_predictor(self.img, rect)
        points = [(el.x, el.y) for el in list(pose_landmarks.parts())]
        return points

    def run(self, predict_face=False):
        print('Start')
        points = self.predict_landmarks(predict_face)
        self.land_to_txt(points)
        print('Landmarks saved!')


if __name__ == '__main__':
    lm = LandmarksDetector(land_path=LAND_PATH,
                           img_path=IMG_PATH,
                           file_name=FILE_NAME)
    lm.run(predict_face=True)
