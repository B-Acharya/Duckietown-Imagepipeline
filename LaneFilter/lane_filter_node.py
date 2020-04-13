#!/usr/bin/env python
#from cv_bridge import CvBridge
import numpy as np
import cv2
from .lane_filter import LaneFilterHistogram
import json
import time
class LaneFilterNode(object):

    def __init__(self, segmentlst):
        self.node_name = "Lane Filter"
        self.active = True
        self.filter = LaneFilterHistogram()
        #actual code consists of a message too get the current velocity and angle
        self.velocity = 0.230000004172
        self.omega = 0.363478302956

        self.d_median = []
        self.phi_median = []
        self.latencyArray = []
        self.t_last_update = time.time()

        # Define Constants
        self.curvature_res = self.filter.curvature_res
        self.processSegments(segmentlst)
        # self.pub_in_lane    = rospy.Publisher("~in_lane",BoolStamped, queue_size=1)
        # Subscribers
        #self.sub = rospy.Subscriber("~segment_list", SegmentList, self.processSegments, queue_size=1)
        # self.sub_velocity = rospy.Subscriber("~car_cmd", Twist2DStamped, self.updateVelocity)
        # self.sub_change_params = rospy.Subscriber("~change_params", String, self.cbChangeParams)
        # Publishers
        # self.pub_lane_pose = rospy.Publisher("~lane_pose", LanePose, queue_size=1)
        # self.pub_belief_img = rospy.Publisher("~belief_img", Image, queue_size=1)
        # self.pub_seglist_filtered = rospy.Publisher("~seglist_filtered",SegmentList, queue_size=1)
        #
        # self.pub_ml_img = rospy.Publisher("~ml_img", Image, queue_size=1)
        #
        #
        # self.pub_entropy    = rospy.Publisher("~entropy",Float32, queue_size=1)
        #
        #
        # FSM
        # self.sub_switch = rospy.Subscriber("~switch",BoolStamped, self.cbSwitch, queue_size=1)
        # self.sub_fsm_mode = rospy.Subscriber("~fsm_mode", FSMState, self.cbMode, queue_size=1)
        # self.active = True
        #
        # timer for updating the params
        # self.timer = rospy.Timer(rospy.Duration.from_sec(1.0), self.updateParams)

    def processSegments(self,segment_list_msg):
        # Get actual timestamp for latency measurement
        timestamp_now = time.time()


        if not self.active:
            return

        # TODO-TAL double call to param server ... --> see TODO in the readme, not only is it never updated, but it is alwas 0
        # Step 0: get values from server
        # if (rospy.get_param('~curvature_res') is not self.curvature_res):
        #     self.curvature_res = rospy.get_param('~curvature_res')
        #     self.filter.updateRangeArray(self.curvature_res)

        # Step 1: predict
        current_time = time.time()
        dt = current_time - self.t_last_update
        v = self.velocity
        w = self.omega

        self.filter.predict(dt=dt, v=v, w=w)
        self.t_last_update = current_time

        # Step 2: update

        self.filter.update(segment_list_msg)

        # Step 3: build messages and publish things
        [d_max, phi_max] = self.filter.getEstimate()
        # print "d_max = ", d_max
        # print "phi_max = ", phi_max
        inlier_segments_msg = []
        inlier_segments = self.filter.get_inlier_segments(segment_list_msg, d_max, phi_max)
        # inlier_segments_msg = SegmentList()
        # inlier_segments_msg.header = segment_list_msg.header
        inlier_segments_msg.append(inlier_segments)
        # self.pub_seglist_filtered.publish(inlier_segments_msg)


        max_val = self.filter.getMax()
        in_lane = max_val > self.filter.min_max
        # build lane pose message to send
        lanePose = {"d": d_max[0], "phi": phi_max[0], "in_lane": in_lane, "status": "NORMAL"}
        # lanePose.header.stamp = segment_list_msg.header.stamp
        # XXX: is it always NORMAL?

        if self.curvature_res > 0:
            lanePose.curvature = self.filter.getCurvature(d_max[1:], phi_max[1:])

        # self.pub_lane_pose.publish(lanePose)

        # TODO-TAL add a debug param to not publish the image !!
        # TODO-TAL also, the CvBridge is re instantiated every time... 
        # publish the belief image
        # bridge = CvBridge()
        belief_img =np.array(255 * self.filter.beliefArray[0]).astype("uint8")
        # belief_img.header.stamp = segment_list_msg.header.stamp
        cv2.imshow("belief_image", belief_img)
        
        
        
        # Latency of Estimation including curvature estimation
        estimation_latency_stamp = time.time() - timestamp_now
        estimation_latency = estimation_latency_stamp + estimation_latency_stamp
        self.latencyArray.append(estimation_latency)

        if (len(self.latencyArray) >= 20):
            self.latencyArray.pop(0)

        # print "Latency of segment list: ", segment_latency
        # print("Mean latency of Estimation:................. %s" % np.mean(self.latencyArray))

        # also publishing a separate Bool for the FSM
        in_lane_msg = {}
        # in_lane_msg.header.stamp = segment_list_msg.header.stamp
        in_lane_msg["data"] = True #TODO-TAL change with in_lane. Is this messqge useful since it is alwas true ?
        # self.pub_in_lane.publish(in_lane_msg)


if __name__ == '__main__':
    #rospy.init_node('lane_filter', anonymous=False)
    lane_filter_node = LaneFilterNode()
    #rospy.on_shutdown(lane_filter_node.onShutdown)
    #rospy.spin()
