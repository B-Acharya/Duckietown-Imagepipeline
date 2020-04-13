from ProcessImage.ProcessImage import ProcessImage
from GroundProjection.ground_projection_geometry import GroundProjection
from ProcessImage.line_detector.line_detector_plot import drawLines
from LaneFilter.lane_filter_node import LaneFilterNode
import cv2
import numpy  as np


def get_normalized_points(lines):
    image_size = [120, 160]
    top_cutoff = 40
    arr_cutoff = np.array((0, top_cutoff, 0, top_cutoff))
    arr_ratio = np.array((1. / image_size[1], 1. / image_size[0], 1. / image_size[1], 1. / image_size[0]))
    if len(lines) > 0:
        return (lines + arr_cutoff) * arr_ratio

def get_segments(ColorLines, normals, color):
    segmentMsgList = []
    for x1, y1, x2, y2, norm_x, norm_y in np.hstack((ColorLines, normals)):
        segment = {}
        segment["color"] = color
        segment["pixels_normalized_start"] = np.array([x1, y1])
        segment["pixels_normalized_end"] = np.array([x2, y2])
        segment["normal_x"] = norm_x
        segment["normal_y"] = norm_y
        segmentMsgList.append(segment)
    return segmentMsgList

def seg_gp(seglist_msg):
    seglist_out = []
    for received_segment in seglist_msg:
        new_segment = {}
        new_segment["pixels_normalized_start"] = gpg.vector2ground(received_segment["pixels_normalized_start"])
        new_segment["pixels_normalized_end"] = gpg.vector2ground(received_segment["pixels_normalized_end"])
        new_segment["color"] = received_segment["color"]
        # TODO what about normal and points
        seglist_out.append(new_segment)
    return seglist_out


if __name__ == '__main__':
    process_image = ProcessImage("image4.png")
    gpg = GroundProjection()
    segments_white = get_segments(get_normalized_points(process_image.white.lines), process_image.white.normals, "white")
    segments_yellow = get_segments(get_normalized_points(process_image.yellow.lines), process_image.yellow.normals, "yellow")
    gp_segments_white = seg_gp(segments_white)
    gp_segments_yellow = seg_gp(segments_yellow)
    segmentlst = []
    segmentlst.extend(gp_segments_white)
    segmentlst.extend(gp_segments_yellow)
    LaneFilterNode(segmentlst)




    cv2.imshow("image with lines", process_image.image_with_lines)
    cv2.imshow("color segment", process_image.colorSegment)
    cv2.imshow("edge ", process_image.edge)
    cv2.waitKey(0)
