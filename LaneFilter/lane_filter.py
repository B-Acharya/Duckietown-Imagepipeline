from collections import OrderedDict
import matplotlib.pyplot as plt
from math import floor

from numpy.testing.utils import assert_almost_equal
from scipy.ndimage.filters import gaussian_filter
from scipy.stats import multivariate_normal, entropy

import numpy as np
import math

# from .visualization import plot_phi_d_diagram_bgr

from scipy.stats import multivariate_normal
from scipy.ndimage.filters import gaussian_filter
from math import floor, sqrt, ceil
import copy


class LaneFilterHistogram:
    # """LaneFilterHistogram"""

    def __init__(self):
        self.mean_d_0 = 0
        self.mean_phi_0 = 0
        self.sigma_d_0 = 0.1
        self.sigma_phi_0 = 0.1
        self.delta_d = 0.02
        self.delta_phi = 0.1
        self.d_max = 0.3
        self.d_min = -0.15
        self.phi_min = -1.5
        self.phi_max = 1.5
        self.cov_v = 0.5
        self.linewidth_white = 0.05
        self.linewidth_yellow = 0.025
        self.lanewidth = 0.23
        self.min_max = 0.1
        self.sigma_d_mask = 1.0
        self.sigma_phi_mask = 2.0
        self.curvature_res = 0
        self.range_min = 0.2
        self.range_est = 0.33
        self.range_max = 0.6
        self.curvature_right = -0.054
        self.curvature_left = 0.025

        self.d, self.phi = np.mgrid[self.d_min:self.d_max:self.delta_d,
                           self.phi_min:self.phi_max:self.delta_phi]
        self.d_pcolor, self.phi_pcolor = \
            np.mgrid[self.d_min:(self.d_max + self.delta_d):self.delta_d,
            self.phi_min:(self.phi_max + self.delta_phi):self.delta_phi]

        self.beliefArray = []
        self.range_arr = np.zeros(self.curvature_res + 1)
        for i in range(self.curvature_res + 1):
            self.beliefArray.append(np.empty(self.d.shape))
        self.mean_0 = [self.mean_d_0, self.mean_phi_0]
        self.cov_0 = [[self.sigma_d_0, 0], [0, self.sigma_phi_0]]
        self.cov_mask = [self.sigma_d_mask, self.sigma_phi_mask]

        self.d_med_arr = []
        self.phi_med_arr = []
        self.median_filter_size = 5

        self.initialize()
        self.updateRangeArray(self.curvature_res)

        # Additional variables
        self.red_to_white = False
        self.use_yellow = True
        self.range_est_min = 0
        self.filtered_segments = []
        self.segm = []

    def get_entropy(self):
        belief = self.beliefArray[0]
        s = entropy(belief.flatten())
        return s

    # master18 different stuff
    # def initialize(self):
    #     self.beliefArray = []
    #     for _ in range(self.num_belief):
    #         n = self.d.shape[0] * self.d.shape[1]
    #         b = np.ones(self.d.shape) * (1.0 / n)
    #         self.beliefArray.append(b)
    #
    #     pos = np.empty(self.d.shape + (2,))
    #     pos[:, :, 0] = self.d
    #     pos[:, :, 1] = self.phi
    #     # XXX: statement with no effect
    #     # self.cov_0
    #     RV = multivariate_normal(self.mean_0, self.cov_0)
    #
    #     n = pos.shape[0] * pos.shape[1]
    #
    #     gaussian = RV.pdf(pos) * 0.5  #+ 0.5/n
    #
    #     gaussian = gaussian / np.sum(gaussian.flatten())
    #
    #     uniform = np.ones(dtype='float32', shape=self.d.shape) * (1.0 / n)
    #
    #     a = 0.01
    #     self.belief = a * gaussian + (1 - a) * uniform
    #
    #     assert_almost_equal(self.belief.flatten().sum(), 1.0)
    #

    def predict(self, dt, v, w):
        delta_t = dt
        d_t = self.d + v * delta_t * np.sin(self.phi)
        phi_t = self.phi + w * delta_t

        for k in range(self.curvature_res):
            p_belief = np.zeros(self.beliefArray[k].shape)

            # there has got to be a better/cleaner way to do this - just applying the process model to translate each cell value
            for i in range(self.beliefArray[k].shape[0]):
                for j in range(self.beliefArray[k].shape[1]):
                    if self.beliefArray[k][i, j] > 0:
                        if d_t[i, j] > self.d_max or d_t[i, j] < self.d_min or phi_t[i, j] < self.phi_min or phi_t[
                            i, j] > self.phi_max:
                            continue

                        i_new = int(
                            floor((d_t[i, j] - self.d_min) / self.delta_d))
                        j_new = int(
                            floor((phi_t[i, j] - self.phi_min) / self.delta_phi))

                        p_belief[i_new, j_new] += self.beliefArray[k][i, j]

            s_belief = np.zeros(self.beliefArray[k].shape)
            gaussian_filter(p_belief, self.cov_mask,
                            output=s_belief, mode='constant')

            if np.sum(s_belief) == 0:
                return
            self.beliefArray[k] = s_belief / np.sum(s_belief)

    # prepare the segments for the creation of the belief arrays
    def prepareSegments(self, segments):
        segmentsRangeArray = list(map(list, [[]] * (self.curvature_res + 1)))
        self.filtered_segments = []
        for segment in segments:
            # Optional transform from RED to WHITE
            if self.red_to_white and segment["color"] == "white":
                segment["color"] = "white"

            # Optional filtering out YELLOW
            if not self.use_yellow and segment["color"] == "yellow":
                continue

            # we don't care about RED ones for now
            if segment["color"] != "white" and segment["color"] != "yellow":
                continue
            # filter out any segments that are behind us
            if segment["pixels_normalized_start"][0] < 0 or segment["pixels_normalized_end"][0] < 0:
                continue

            self.filtered_segments.append(segment)
            # only consider points in a certain range from the Duckiebot for the position estimation
            point_range = self.getSegmentDistance(segment)
            if point_range < self.range_est and point_range > self.range_est_min:
                segmentsRangeArray[0].append(segment)
                # print functions to help understand the functionality of the code
                # print 'Adding segment to segmentsRangeArray[0] (Range: %s < 0.3)' % (point_range)
                # print 'Printout of last segment added: %s' % self.getSegmentDistance(segmentsRangeArray[0][-1])
                # print 'Length of segmentsRangeArray[0] up to now: %s' % len(segmentsRangeArray[0])
            # split segments ot different domains for the curvature estimation
            if self.curvature_res is not 0:

                for i in range(self.curvature_res):
                    if point_range < self.range_arr[i + 1] and point_range > self.range_arr[i]:
                        segmentsRangeArray[i + 1].append(segment)
                        # print functions to help understand the functionality of the code
                        # print 'Adding segment to segmentsRangeArray[%i] (Range: %s < %s < %s)' % (i + 1, self.range_arr[i], point_range, self.range_arr[i + 1])
                        # print 'Printout of last segment added: %s' % self.getSegmentDistance(segmentsRangeArray[i + 1][-1])
                        # print 'Length of segmentsRangeArray[%i] up to now: %s' % (i + 1, len(segmentsRangeArray[i + 1]))
                        continue
            self.segm.append([segment, False])
        # print functions to help understand the functionality of the code
        # for i in range(len(segmentsRangeArray)):
        #     print 'Length of segmentsRangeArray[%i]: %i' % (i, len(segmentsRangeArray[i]))
        #     for j in range(len(segmentsRangeArray[i])):
        #         print 'Range of segment %i: %f' % (j, self.getSegmentDistance(segmentsRangeArray[i][j]))

        return segmentsRangeArray

    def updateRangeArray(self, curvature_res):
        self.curvature_res = curvature_res
        self.beliefArray = []
        for i in range(self.curvature_res + 1):
            self.beliefArray.append(np.empty(self.d.shape))
        self.initialize()
        if curvature_res > 1:
            self.range_arr = np.zeros(self.curvature_res + 1)
            range_diff = (self.range_max - self.range_min) / \
                         (self.curvature_res)

            # populate range_array
            for i in range(len(self.range_arr)):
                self.range_arr[i] = self.range_min + (i * range_diff)

    # generate the belief arrays
    def update(self, segments , no, data):
        # prepare the segments for each belief array
        segmentsRangeArray = self.prepareSegments(segments)
        # generate all belief arrays
        for i in range(self.curvature_res + 1):
            measurement_likelihood = self.generate_measurement_likelihood(segmentsRangeArray[i], no, data)

            if measurement_likelihood is not None:
                self.beliefArray[i] = np.multiply(self.beliefArray[i], measurement_likelihood)
                if np.sum(self.beliefArray[i]) == 0:
                    self.beliefArray[i] = measurement_likelihood
                else:
                    self.beliefArray[i] = self.beliefArray[i] / np.sum(self.beliefArray[i])

    def generate_measurement_likelihood(self, segments, no, data):

        # initialize measurement likelihood to all zeros
        ##shape for normal
        measurement_likelihood = np.zeros(self.d.shape)
        ## shape for plotting
        measurement_likelihood_1 = np.zeros((30,23))
        seg = []
        k = 0

        for segment in segments:
            d_i, phi_i, l_i, weight = self.generateVote(segment)

            # if the vote lands outside of the histogram discard it
            if d_i > self.d_max or d_i < self.d_min or phi_i < self.phi_min or phi_i > self.phi_max:
                self.segm.append([segment, False])
                continue
            self.segm.append([segment, True])

            ### Mesurement_likelihood update with d in y axis and phi in x axis
            i = int(floor((d_i - self.d_min) / self.delta_d))
            j = int(floor((phi_i - self.phi_min) / self.delta_phi))
            measurement_likelihood[i, j] = measurement_likelihood[i, j] + 1

            ### Mesurement_likelihood update with d in y axis and phi in x axis
            m = int(floor((self.d_max - d_i) / self.delta_d))
            n = int(floor((phi_i - self.phi_min) / self.delta_phi))
            measurement_likelihood_1[n, m] = measurement_likelihood_1[n, m] + 1
            seg.append([segment["pixels_normalized_start"][0],segment["pixels_normalized_end"][0],segment["pixels_normalized_start"][1],segment["pixels_normalized_end"][1]])
            # seg_1 = np.array(seg)
            ### Used to plot both the plots simultaneously
        plt.subplot(211)
        plt.pcolormesh(measurement_likelihood_1)
        plt.yticks(np.arange(0, 33, step=3), np.around(np.arange(self.phi_min*180/math.pi, self.phi_max*180/math.pi + self.delta_phi*180/math.pi*3, step=self.delta_phi*180/math.pi*3),decimals=1))
        plt.xticks(np.arange(0, 23, step=2), np.around(np.arange(0.3, -0.15-0.02*2, step=-0.02*2), decimals=2))
        column = int(floor((self.d_max - data[no][0]) / self.delta_d))
        row = int(floor((data[no][1] - self.phi_min) / self.delta_phi))
        plt.plot(row, column, "ro")
        plt.plot((-1, 1), (0, 0), 'r:')
        plt.plot((0, 0), (-1, 1), 'r:')
        # plt.yticks(np.arange(0, 23, step=3),[0.3, 0.24, 0.18, 0.12, 0.06, 0, 0.06, 0.12 ]
        plt.colorbar()
        plt.subplot(212)
        plt.xlim(1, -1)
        plt.ylim(0, 0.5)
        plt.ylabel("X-axis")
        plt.xlabel("Y-axis")
        # plt.plot([segment["pixels_normalized_start"][0],
        #           segment["pixels_normalized_end"][0]],
        #          [segment["pixels_normalized_start"][1],
        #           segment["pixels_normalized_end"][1]])
        for s in seg:
            plt.plot(s[2:4], s[0:2])
            # plt.show()
            k += 1
        maxids = np.unravel_index(
            measurement_likelihood_1.argmax(), measurement_likelihood_1.shape)
        d_max = self.d_max - (maxids[0]) * self.delta_d
        plt.plot(d_max, 0, 'ro')
        plt.show()
        # plt.savefig('Image_{}_fig_{}.png'.format(no, k))

        plt.clf()
        # for segs in self.segm:
        #     plt.xlim(1, -1)
        #     plt.ylim(0, 1)
        #     if segs[1] == True:
        #         plt.plot([segs[0]["pixels_normalized_start"][1],
        #                   segs[0]["pixels_normalized_end"][1]],[segs[0]["pixels_normalized_start"][0],
        #                   segs[0]["pixels_normalized_end"][0]])
        #     else:
        #         plt.plot([segs[0]["pixels_normalized_start"][1], segs[0]["pixels_normalized_end"][1]],
        #                  [segs[0]["pixels_normalized_start"][0],segs[0]["pixels_normalized_end"][0]], color='red', linestyle='dashed')
        # plt.savefig("line_plot.png")
        if np.linalg.norm(measurement_likelihood) == 0:
            return None
        measurement_likelihood = measurement_likelihood / \
                                 np.sum(measurement_likelihood)
        return measurement_likelihood

    # get the maximal values d_max and phi_max from the belief array. The first belief array (beliefArray[0]) includes the actual belief of the Duckiebots position. The further belief arrays are used for the curvature estimation.
    def getEstimate(self):
        d_max = np.zeros(self.curvature_res + 1)
        phi_max = np.zeros(self.curvature_res + 1)
        for i in range(self.curvature_res + 1):
            maxids = np.unravel_index(
                self.beliefArray[i].argmax(), self.beliefArray[i].shape)
            d_max[i] = self.d_min + (maxids[0] + 0.5) * self.delta_d
            phi_max[i] = self.phi_min + (maxids[1] + 0.5) * self.delta_phi
        return [d_max, phi_max]

    def get_estimate(self):
        d, phi = self.getEstimate()
        res = OrderedDict()
        res['d'] = d[0]
        res['phi'] = phi[0]
        return res

    # get the curvature estimation
    def getCurvature(self, d_max, phi_max):
        d_med_act = np.median(d_max)  # actual median d value
        phi_med_act = np.median(phi_max)  # actual median phi value

        # store median values over a few time step to smoothen out the estimation
        if len(self.d_med_arr) >= self.median_filter_size:
            self.d_med_arr.pop(0)
            self.phi_med_arr.pop(0)
        self.d_med_arr.append(d_med_act)
        self.phi_med_arr.append(phi_med_act)

        # set curvature
        # TODO: check magic constants
        if np.median(self.phi_med_arr) - phi_max[0] < -0.3 and np.median(self.d_med_arr) > 0.05:
            print("Curvature estimation: left curve")
            return self.curvature_left
        elif np.median(self.phi_med_arr) - phi_max[0] > 0.2 and np.median(self.d_med_arr) < -0.02:
            print("Curvature estimation: right curve")
            return self.curvature_right
        else:
            print("Curvature estimation: straight lane")
            return 0

    # return the maximal value of the beliefArray
    def getMax(self):
        return self.beliefArray[0].max()

    def initialize(self):
        pos = np.empty(self.d.shape + (2,))
        pos[:, :, 0] = self.d
        pos[:, :, 1] = self.phi
        self.cov_0
        RV = multivariate_normal(self.mean_0, self.cov_0)
        for i in range(self.curvature_res + 1):
            self.beliefArray[i] = RV.pdf(pos)
        # plt.pyplot.contourf(self.d, self.phi, RV.pdf(pos))
        # plt.pyplot.show()

    # generate a vote for one segment
    def generateVote(self, segment):
        p1 = segment["pixels_normalized_start"][0:2]  # np.array([segment.points[0].x, segment.points[0].y])
        p2 = segment["pixels_normalized_end"][0:2]  # np.array([segment.points[1].x, segment.points[1].y])
        t_hat = (p2 - p1) / np.linalg.norm(p2 - p1)

        n_hat = np.array([-t_hat[1], t_hat[0]])
        d1 = np.inner(n_hat, p1)
        d2 = np.inner(n_hat, p2)
        l1 = np.inner(t_hat, p1)
        l2 = np.inner(t_hat, p2)
        if (l1 < 0):
            l1 = -l1
        if (l2 < 0):
            l2 = -l2

        l_i = (l1 + l2) / 2
        d_i = (d1 + d2) / 2
        phi_i = np.arcsin(t_hat[1])
        if segment["color"] == "white":  # right lane is white
            if (p1[0] > p2[0]):  # right edge of white lane
                d_i = d_i - self.linewidth_white
            else:  # left edge of white lane

                d_i = - d_i

                phi_i = -phi_i
            d_i = d_i - self.lanewidth / 2

        elif segment["color"] == "yellow":  # left lane is yellow
            if (p2[0] > p1[0]):  # left edge of yellow lane
                d_i = d_i - self.linewidth_yellow
                phi_i = -phi_i
            else:  # right edge of white lane
                d_i = -d_i
            d_i = self.lanewidth / 2 - d_i

        # weight = distance
        weight = 1
        return d_i, phi_i, l_i, weight

    def get_inlier_segments(self, segments, d_max, phi_max):
        inlier_segments = []
        for segment in segments:
            d_s, phi_s, l, w = self.generateVote(segment)
            if abs(d_s - d_max) < self.delta_d and abs(phi_s - phi_max) < self.delta_phi:
                inlier_segments.append(segment)
        return inlier_segments

    # get the distance from the center of the Duckiebot to the center point of a segment
    def getSegmentDistance(self, segment):
        x_c = (segment["pixels_normalized_start"][0] + segment["pixels_normalized_end"][0]) / 2
        y_c = (segment["pixels_normalized_start"][1] + segment["pixels_normalized_end"][1]) / 2
        return sqrt(x_c ** 2 + y_c ** 2)
    #
    # def get_plot_phi_d(self, ground_truth=None):  # @UnusedVariable
    #     d, phi = self.getEstimate()
    #     belief = self.beliefArray[0]
    #     return plot_phi_d_diagram_bgr(self, belief, phi=phi, d=d)
