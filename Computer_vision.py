import cv2
import numpy as np
from line_detector.line_detector import LineDetectorHSV
from line_detector.line_detector_plot import drawLines, color_segment, drawNormals


def bgr_image(file):
    return cv2.imread(file)


if __name__ == '__main__':
    image_cv = bgr_image("/home/bhargav/Pictures/duckei_town/image4.png")

    detector_used = LineDetectorHSV()
    cv2.imshow("image",image_cv)

    image_size = [120, 160]
    top_cutoff = 40
    hei_original, wid_original = image_cv.shape[0:2]

    if image_size[0] != hei_original or image_size[1] != wid_original:
        # image_cv = cv2.GaussianBlur(image_cv, (5,5), 2)
        image_cv = cv2.resize(image_cv, (image_size[1], image_size[0]), interpolation=cv2.INTER_NEAREST)

    image_cv = image_cv[top_cutoff:, :, :]

    detector_used.setImage(image_cv)

    white = detector_used.detectLines('white')
    yellow = detector_used.detectLines('yellow')
    red = detector_used.detectLines('red')

    arr_cutoff = np.array((0, top_cutoff, 0, top_cutoff))
    arr_ratio = np.array((1. / image_size[1], 1. / image_size[0], 1. / image_size[1], 1. / image_size[0]))

    if len(white.lines) > 0:
        lines_normalized_white = ((white.lines + arr_cutoff) * arr_ratio)
    if len(yellow.lines) > 0:
        lines_normalized_yellow = ((yellow.lines + arr_cutoff) * arr_ratio)
    if len(red.lines) > 0:
        lines_normalized_red = ((red.lines + arr_cutoff) * arr_ratio)

    image_with_lines = np.copy(image_cv)
    drawLines(image_with_lines, white.lines, (0, 0, 0))
    drawLines(image_with_lines, yellow.lines, (255, 0, 0))
    drawLines(image_with_lines, red.lines, (0, 255, 0))

    drawNormals(image_with_lines, white.lines, white.normals)
    drawNormals(image_with_lines, yellow.lines, yellow.normals)

    colorSegment = color_segment(white.area, red.area, yellow.area)
    edge = detector_used.edges

    cv2.imshow("image with lines", image_with_lines)
    cv2.imshow("color segment", colorSegment)
    cv2.imshow("edge ", edge)
    cv2.waitKey(0)
