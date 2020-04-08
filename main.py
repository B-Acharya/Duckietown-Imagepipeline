from ProcessImage.ProcessImage import ProcessImage
from GroundProjection.ground_projection_geometry import GroundProjection
import cv2
import  numpy  as np
def get_normalized_points(lines):
    image_size = [120, 160]
    top_cutoff = 40
    arr_cutoff = np.array((0, top_cutoff, 0, top_cutoff))
    arr_ratio = np.array((1. / image_size[1], 1. / image_size[0], 1. / image_size[1], 1. / image_size[0]))
    if len(lines) > 0:
        return (lines + arr_cutoff) * arr_ratio


if __name__ == '__main__':
    process_image = ProcessImage("image4.png")
    ground_projected_white = GroundProjection(get_normalized_points(process_image.white.lines))
    ground_projected_yellow = GroundProjection(get_normalized_points(process_image.yellow.lines))
    ground_projected_red = GroundProjection(get_normalized_points(process_image.red.lines))



    cv2.imshow("image with lines", process_image.image_with_lines)
    cv2.imshow("color segment", process_image.colorSegment)
    cv2.imshow("edge ", process_image.edge)
    cv2.waitKey(0)