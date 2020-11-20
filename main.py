from ProcessImage.ProcessImage import ProcessImage
from GroundProjection.ground_projection_geometry import GroundProjection
from ProcessImage.line_detector.line_detector_plot import drawLines
import matplotlib.pyplot as plt
from LaneFilter.lane_filter_node import LaneFilterNode
from lane_controller.lane_controller_node import lane_controller
import cv2
import pandas as pd
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
    point_array = []
    normalized_point = []
    for received_segment in seglist_msg:
        new_segment = {}
        new_segment["pixels_normalized_start"], point_1 = gpg.vector2ground(received_segment["pixels_normalized_start"])
        new_segment["pixels_normalized_end"], point_2 = gpg.vector2ground(received_segment["pixels_normalized_end"])
        new_segment["color"] = received_segment["color"]
        normalized_point.append(new_segment["pixels_normalized_start"])
        normalized_point.append(new_segment["pixels_normalized_end"])
        point_array.append(point_1)
        point_array.append(point_2)
        # TODO what about normal and points
        seglist_out.append(new_segment)
    return seglist_out, point_array,  normalized_point

def line_on_image(segs, image):
    seg_new = []
    seg_1 = []
    j = 0
    for seg in segs:
        plt.xlim(1, -1)
        plt.ylim(0, 1)
        if seg[1] == True:
            # plt.plot([seg[0]["pixels_normalized_start"][1],
            #           seg[0]["pixels_normalized_end"][1]], [seg[0]["pixels_normalized_start"][0],
            #                                                  seg[0]["pixels_normalized_end"][0]])
            a = list(gpg.ground2pixel([seg[0]["pixels_normalized_start"][0], seg[0]["pixels_normalized_start"][1], 0]))
            b = list(gpg.ground2pixel([seg[0]["pixels_normalized_end"][0], seg[0]["pixels_normalized_end"][1], 0]))
            seg_new.append(a+b)
        # else:
            # plt.plot([segs[0]["pixels_normalized_start"][1], segs[0]["pixels_normalized_end"][1]],
            #          [segs[0]["pixels_normalized_start"][0], segs[0]["pixels_normalized_end"][0]], color='red',
            #          linestyle='dashed')
    for a in seg_new:
        drawLines(image, [a], (0, 0, 0))
        j += 1

    cv2.imwrite("/home/bhargav/Duckietown-Imagepipeline/" + process_image.image_name + "_segment{}.jpeg".format(j),
                image)



    return image, j
    # drawLines(image, seg_new, (0, 0, 0))
    # cv2.imshow("line")
    # plt.savefig("line_plot.png")


if __name__ == '__main__':
    df = pd.read_csv("./out_zig_zag.csv")
    df.drop_duplicates(subset="count", keep = 'first', inplace = True)
    data = []
    data_phi = df['phi'].to_list()
    data_di = df['d_i'].to_list()
    for x in range(len(data_di)):
        data.append([data_di[x] , data_phi[x]])
    for a in range(158):
        a = 123
        # if a == 1:
        #     continue
        process_image = ProcessImage("ImagesFrame_{}".format(a))
        gpg = GroundProjection()
        segmentlst = []
        if process_image.white.lines != [] :
            segments_white = get_segments(get_normalized_points(process_image.white.lines), process_image.white.normals, "white")
            gp_segments_white, point_start, normalized_point_white = seg_gp(segments_white)
            segmentlst.extend(gp_segments_white)
        if process_image.yellow.lines != []:
            segments_yellow = get_segments(get_normalized_points(process_image.yellow.lines), process_image.yellow.normals, "yellow")
            gp_segments_yellow, point_end, normalized_point_yellow = seg_gp(segments_yellow)
            segmentlst.extend(gp_segments_yellow)
        if process_image.red.lines != []:
            segments_red = get_segments(get_normalized_points(process_image.red.lines), process_image.red.normals, "red")

        # point_start.extend(point_end)
        # normalized_point_white.extend(normalized_point_yellow)
        # points = np.array(point_start)
        # norms = np.array(normalized_point_white)
        #
        # fig, ax = plt.subplots(2)
        # im = ax[1].scatter(points[:, 0], points[:, 1], c = points[:, 2])
        # fig.colorbar(im)
        # ax[0].scatter(norms[:, 0], norms[:, 1], c = points[:, 2])
        # ax[0].set_title("After division")
        # ax[1].set_title("Before division")
        # plt.show()
        #
        #
        # ##line plotting
        # for segment in gp_segments_white:
        #     plt.plot([segment["pixels_normalized_start"][0], segment["pixels_normalized_end"][0]], [segment["pixels_normalized_start"][1], segment["pixels_normalized_end"][1]])
        # for segment in gp_segments_yellow:
        #     plt.plot([segment["pixels_normalized_start"][0], segment["pixels_normalized_end"][0]],
        #              [segment["pixels_normalized_start"][1], segment["pixels_normalized_end"][1]])
        # plt.show()







        lane_filter = LaneFilterNode(segmentlst, a, data)
        lane_controller_1 = lane_controller(lane_filter.lanePose)

        # call to plot the line
        # img = cv2.imread("/home/bhargav/Duckietown-Imagepipeline/Images/ImagesFrame_{}.png".format(a))##np.copy(process_image.image_cv)
        # img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_NEAREST)
        # final, k = line_on_image(lane_filter.filter.segm, img)
        #
        # # for i in range(k):
        # img1 = cv2.imread('/home/bhargav/Duckietown-Imagepipeline/Image_{}_fig_{}.png'.format(a, k))
        # img2 = cv2.imread('/home/bhargav/Duckietown-Imagepipeline/ImagesFrame_{}_segment{}.jpeg'.format(a, k))
        # con = np.concatenate((img1, img2), axis=0)
        # cv2.imwrite('frame_{}_segments_{}.jpeg'.format(a, k), con)



        # cv2.imshow("image with lines", process_image.image_with_lines)
        # cv2.imshow("color segment", process_image.colorSegment)
        # cv2.imshow("edge ", process_image.edge)
        # cv2.imshow("proccesd image", final)
        # cv2.waitKey(0)
