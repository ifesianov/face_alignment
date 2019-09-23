import cv2
import math
import numpy as np


class MathHelper:
    @staticmethod
    def compute_similarity_transformation(in_points, out_points):
        """
        Compute similarity transform given two sets of two points.
        OpenCV requires 3 pairs of corresponding points.
        We are faking the third one.
        """
        s60 = math.sin(60 * math.pi / 180)
        c60 = math.cos(60 * math.pi / 180)

        in_points_list = np.copy(in_points).tolist()
        out_points_list = np.copy(out_points).tolist()

        x_in = c60 * (in_points_list[0][0] - in_points_list[1][0]) - s60 * (
                    in_points_list[0][1] - in_points_list[1][1]) + in_points_list[1][0]
        y_in = s60 * (in_points_list[0][0] - in_points_list[1][0]) + c60 * (
                    in_points_list[0][1] - in_points_list[1][1]) + in_points_list[1][1]

        in_points_list.append([np.int(x_in), np.int(y_in)])

        x_out = c60 * (out_points_list[0][0] - out_points_list[1][0]) - s60 * (
                    out_points_list[0][1] - out_points_list[1][1]) + out_points_list[1][0]
        y_out = s60 * (out_points_list[0][0] - out_points_list[1][0]) + c60 * (
                    out_points_list[0][1] - out_points_list[1][1]) + out_points_list[1][1]

        out_points_list.append([np.int(x_out), np.int(y_out)])

        transformation = cv2.estimateAffinePartial2D(np.array([in_points_list]), np.array([out_points_list]))[0]

        return transformation

    @staticmethod
    def calculateDelaunayTriangles(rect, points):
        subdiv = cv2.Subdiv2D(rect)
        for p in points:
            subdiv.insert((p[0], p[1]))

        triangles_list = subdiv.getTriangleList()

        # Find the indices of triangles in the points array

        delaunau_tri = []

        for t in triangles_list:
            pt = []
            pt.append((t[0], t[1]))
            pt.append((t[2], t[3]))
            pt.append((t[4], t[5]))

            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])

            if MathHelper.is_in_rect(rect, pt1) and MathHelper.is_in_rect(rect, pt2) and MathHelper.is_in_rect(rect, pt3):
                ind = []
                for j in range(0, 3):
                    for k in range(0, len(points)):
                        if abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0:
                            ind.append(k)
                if len(ind) == 3:
                    delaunau_tri.append((ind[0], ind[1], ind[2]))

        return np.array(delaunau_tri)

    @staticmethod
    def is_in_rect(rect, point):
        if point[0] < rect[0]:
            return False
        elif point[1] < rect[1]:
            return False
        elif point[0] > rect[2]:
            return False
        elif point[1] > rect[3]:
            return False
        return True

    @staticmethod
    def constrain_point(p, w, h):
        p = (min(max(p[0], 0), w - 1), min(max(p[1], 0), h - 1))
        return p

    @staticmethod
    def applyAffineTransform(src, srcTri, dstTri, size):

        # Given a pair of triangles, find the affine transform.
        warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))

        # Apply the Affine Transform just found to the src image
        dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT_101)

        return dst

    @staticmethod
    def warp_triangle(img1, img2, t1, t2):

        # Find bounding rectangle for each triangle
        r1 = cv2.boundingRect(np.float32([t1]))
        r2 = cv2.boundingRect(np.float32([t2]))

        # Offset points by left top corner of the respective rectangles
        t1Rect = []
        t2Rect = []
        t2RectInt = []

        for i in range(0, 3):
            t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
            t2Rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
            t2RectInt.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

        # Get mask by filling triangle
        mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0)

        # Apply warpImage to small rectangular patches
        img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]

        size = (r2[2], r2[3])

        img2Rect = MathHelper.applyAffineTransform(img1Rect, t1Rect, t2Rect, size)

        img2Rect = img2Rect * mask

        # Copy triangular region of the rectangular patch to the output image
        img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] * (
                    (1.0, 1.0, 1.0) - mask)

        img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] + img2Rect