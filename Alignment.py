import os
import cv2
import numpy as np
from File import File
from MathHelper import MathHelper

IMG_PATH = 'images'
LAND_PATH = 'landmarks'
DEFORMED_IMG_PATH = 'deformed_images'

FILE_NAME = 'atynkaliuk.jpg'


class Alignment:
    # Dimensions of output image
    W = 600
    H = 600

    def __init__(self, land_path, img_path, deformed_img_path, ideal_mask, file_name):
        self.land_path = land_path
        self.img_path = img_path
        self.deformed_img_path = deformed_img_path

        self.img_file_name = file_name
        self._land_file_name = None
        self._deformed_img_file_name = None

        self._triangulation = None
        self.ideal_mask = ideal_mask

        self._img = None
        self._land = None
        self._boundary_points = None

        self._norm_img = None
        self._norm_land = None

        self.deformed_img = np.zeros((self.H, self.W, 3), np.float32())

    def run(self):
        # print('Alignment of "{}" has been started'.format(self.img_file_name))
        self.warp()
        self.crop_img()
        self.save_deformed_img()
        # self.show_deformed_img()
        # print('Done')
        # print('Alignmented image is saved.')

    @property
    def land_file_name(self):
        if self._land_file_name is None:
            self._land_file_name = os.path.splitext(self.img_file_name)[0] + '.txt'
        return self._land_file_name

    @property
    def deformed_img_file_name(self):
        if self._deformed_img_file_name is None:
            self._deformed_img_file_name = 'deformed_{}'.format(self.img_file_name)
        return self._deformed_img_file_name

    @property
    def norm_img(self):
        if self._norm_img is None:
            self._norm_img, self._norm_land = self.eyes_similarity_transformation()
        return self._norm_img

    @property
    def norm_land(self):
        if self._norm_land is None:
            self._norm_img, self._norm_land = self.eyes_similarity_transformation()
        return self._norm_land

    @property
    def img(self):
        if self._img is None:
            self._img = File.read_img(self.img_path, self.img_file_name, convert_to_float=True)
        return self._img

    @property
    def land(self):
        if self._land is None:
            self._land = File.read_land(self.land_path, self.land_file_name)
        return self._land

    @property
    def triangulation(self):
        if self._triangulation is None:
            rect = (0, 0, self.W, self.H)
            self._triangulation = MathHelper.calculateDelaunayTriangles(rect, np.array(self.ideal_mask))
        return self._triangulation

    @property
    def boundary_points(self):
        """
        Add boundary points for Delaunay triangulation
        """
        if self._boundary_points is None:
            self._boundary_points = np.array([(0, 0),
                                              (self.W / 2, 0),
                                              (self.W - 1, 0),
                                              (self.W - 1, self.H / 2),
                                              (self.W - 1, self.H - 1),
                                              (self.W / 2, self.H - 1),
                                              (0, self.H - 1),
                                              (0, self.H / 2)])
        return self._boundary_points

    def eyes_similarity_transformation(self):
        eyes_corners = self.get_eyes_corners_points()
        ideal_eyes_corners = self.calc_ideal_eyes_corners()
        return self.similarity_transformation(eyes_corners, ideal_eyes_corners)

    def calc_ideal_eyes_corners(self):
        return [(np.int(0.3 * self.W),
                 np.int(self.H / 3)),
                (np.int(0.7 * self.W),
                 np.int(self.H / 3))]

    def get_eyes_corners_points(self):
        return [self.land[36],
                self.land[45]]

    def similarity_transformation(self, in_points, out_points):

        transformation = MathHelper.compute_similarity_transformation(in_points, out_points)

        norm_img = cv2.warpAffine(self.img, transformation, (self.W, self.H))
        reshaped_land = np.reshape(np.array(self.land), (68, 1, 2))

        norm_land = cv2.transform(reshaped_land, transformation)
        norm_land = np.float32(np.reshape(norm_land, (68, 2)))

        # Append boundary points. Will be used in Delaunay Triangulation
        norm_land = np.append(norm_land, self.boundary_points, axis=0)

        return norm_img, norm_land

    def warp(self):
        for j in range(len(self.triangulation)):
            t_in = []
            t_out = []

            for k in range(3):
                vertex = self.triangulation[j][k]

                p_in = self.norm_land[vertex]
                p_in = MathHelper.constrain_point(p_in, self.W, self.H)

                p_out = self.ideal_mask[vertex]
                p_out = MathHelper.constrain_point(p_out, self.W, self.H)

                t_in.append(p_in)
                t_out.append(p_out)

            MathHelper.warp_triangle(self.norm_img, self.deformed_img, t_in, t_out)

    def crop_img(self):
        x0 = int(self.ideal_mask[1][0])
        x1 = int(self.ideal_mask[15][0])
        y0 = int(self.ideal_mask[19][1])
        y1 = int(self.ideal_mask[8][1])

        self.deformed_img = self.deformed_img[y0:y1, x0:x1]

    def show_deformed_img(self):
        File.show_img(self.deformed_img)

    def save_deformed_img(self):
        File.save_img(os.path.join(self.deformed_img_path, self.deformed_img_file_name), self.deformed_img, convert_to_float=True)


