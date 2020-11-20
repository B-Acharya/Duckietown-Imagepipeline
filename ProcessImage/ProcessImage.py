import cv2
import os
import numpy as np
from ProcessImage.line_detector.line_detector import LineDetectorHSV
from ProcessImage.line_detector.line_detector_plot import drawLines, color_segment, drawNormals
import pandas as pd

class ProcessImage:

    def __init__(self, file):
        # self.file = file
        self.image_cv = self.BgrImage(file)
        # cv2.imshow("image",self.image_cv)
        # cv2.waitKey(0)
        # self.image_cv = pd.DataFrame.to_numpy(file)
        self.image_name  = file

        self.detector_used = LineDetectorHSV()
        # cv2.imshow("image", image_cv)
        image_size = [120, 160]
        top_cutoff = 40
        hei_original, wid_original = self.image_cv.shape[0:2]

        if image_size[0] != hei_original or image_size[1] != wid_original:
            # image_cv = cv2.GaussianBlur(image_cv, (5,5), 2)
            self.image_cv = cv2.resize(self.image_cv, (image_size[1], image_size[0]), interpolation=cv2.INTER_NEAREST)

        self.image_cv = self.image_cv[top_cutoff:, :, :]
        # cv2.imshow("resized",self.image_cv)
        # cv2.waitKey(0)
        self.detector_used.setImage(self.image_cv)

        self.white = self.detector_used.detectLines('white')
        self.yellow = self.detector_used.detectLines('yellow')
        self.red = self.detector_used.detectLines('red')

        # arr_cutoff = np.array((0, top_cutoff, 0, top_cutoff))
        # arr_ratio = np.array((1. / image_size[1], 1. / image_size[0], 1. / image_size[1], 1. / image_size[0]))
        #
        # if len(self.white.lines) > 0:
        #     self.lines_normalized_white = ((self.white.lines + arr_cutoff) * arr_ratio)
        # if len(self.yellow.lines) > 0:
        #     self.lines_normalized_yellow = ((self.yellow.lines + arr_cutoff) * arr_ratio)
        # if len(self.red.lines) > 0:
        #     self.lines_normalized_red = ((self.red.lines + arr_cutoff) * arr_ratio)

        self.image_with_lines = np.copy(self.image_cv)
        # drawLines(self.image_with_lines, self.white.lines, (0, 0, 0))
        # drawLines(self.image_with_lines, self.yellow.lines, (255, 0, 0))
        # drawLines(self.image_with_lines, self.red.lines, (0, 255, 0))
        #
        # drawNormals(self.image_with_lines, self.white.lines, self.white.normals)
        # drawNormals(self.image_with_lines, self.yellow.lines, self.yellow.normals)

        self.colorSegment = color_segment(self.white.area, self.red.area, self.yellow.area)
        self.edge = self.detector_used.edges

    def BgrImage(self, file):
        path = os.path.abspath('.')
        return cv2.imread(path + "/Images/" + file + ".png")


