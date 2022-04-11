#!/usr/bin/env python
import os
from collections import deque
from data_utils import *
from publish_utils import *
from kitti_util import *

DATA_PATH = '/home/dxy/kitti/rawdata/2011_09_26/2011_09_26_drive_0005_sync/'

def compute_3d_box_cam2(h,w,l,x,y,z,yaw):
    R = np.array([[np.cos(yaw),0,np.sin(yaw)],[0,1,0],[-np.sin(yaw),0,np.cos(yaw)]])
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2]
    y_corners = [0,0,0,0,-h,-h,-h,-h]
    z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]
    corners_3d_cam2 = np.dot(R,np.vstack([x_corners,y_corners,z_corners]))
    corners_3d_cam2 += np.vstack([x,y,z])
    return corners_3d_cam2

class Object():
    def __init__(self):
	self.locations = []

    def update(self,displacement,yaw_change):
	for i in range(len(self.locations)):
	    x0,y0 = self.locations[i]
	    x1 = x0*np.cos(yaw_change)+y0*np.sin(yaw_change)-displacement
	    y1 = -x0*np.sin(yaw_change)+y0*np.cos(yaw_change)
	    self.locations[i] = np.array([x1,y1])
	self.locations += [np.array([0,0])]

    def reset(self):
	self.locations = []

if __name__ == '__main__':
    frame = 0
    rospy.init_node('kitti_node',anonymous=True)
    cam_pub = rospy.Publisher('kitti_cam',Image,queue_size=10)
    pcl_pub = rospy.Publisher('kitti_point_cloud',PointCloud2,queue_size=10)
    ego_pub = rospy.Publisher('kitti_ego_car',Marker,queue_size=10)
    model_pub = rospy.Publisher('kitti_car_model',Marker,queue_size=10)
    imu_pub = rospy.Publisher('kitti_imu',Imu,queue_size=10)
    gps_pub = rospy.Publisher('kitti_gps',NavSatFix,queue_size=10)
    box3d_pub = rospy.Publisher('kitti_3d',MarkerArray,queue_size=10)
    loc_pub = rospy.Publisher('kitti_loc',MarkerArray,queue_size=10)
    bridge = CvBridge()

    rate = rospy.Rate(10)

    df_tracking = read_tracking('/home/dxy/kitti/tracking/training/label_02/0000.txt')
    calib = Calibration('/home/dxy/kitti/rawdata/2011_09_26/',from_video=True)

    ego_car = Object()
    prev_imu_data = None

    while not rospy.is_shutdown():
        df_tracking_frame = df_tracking[df_tracking.frame==frame]

	boxes_2d = np.array(df_tracking_frame[['bbox_left','bbox_top','bbox_right','bbox_bottom']])
	types = np.array(df_tracking_frame['type'])
	boxes_3d = np.array(df_tracking_frame[['height','width','length','pos_x','pos_y','pos_z','rot_y']])
	track_ids = np.array(df_tracking_frame['track_id'])

	corners_3d_velos = []
	for box_3d in boxes_3d:
	    corners_3d_cam2 = compute_3d_box_cam2(*box_3d)
	    corners_3d_velo = calib.project_rect_to_velo(corners_3d_cam2.T)
	    corners_3d_velos += [corners_3d_velo]

	image = read_camera(os.path.join(DATA_PATH,'image_02/data/%010d.png'%frame))
	point_cloud = read_point_cloud(os.path.join(DATA_PATH,'velodyne_points/data/%010d.bin'%frame))
	imu_data = read_imu(os.path.join(DATA_PATH,'oxts/data/%010d.txt'%frame))

	if prev_imu_data is not None:
	    displacement = 0.1*np.linalg.norm(imu_data[['vf','vl']])
	    yaw_change = float(imu_data.yaw-prev_imu_data.yaw)
	    ego_car.update(displacement,yaw_change)
	prev_imu_data = imu_data

	publish_camera(cam_pub,bridge,image,boxes_2d,types)
	publish_point_cloud(pcl_pub,point_cloud)
	publish_3dbox(box3d_pub,corners_3d_velos,types,track_ids)
	publish_ego_car(ego_pub)
	publish_car_model(model_pub)
	publish_imu(imu_pub,imu_data)
	publish_gps(gps_pub,imu_data)
	publish_loc(loc_pub,ego_car.locations)
		
	rospy.loginfo("published frame %d"%frame)
	rate.sleep()
	frame += 1
	if frame == 154:
	    frame = 0
	    ego_car.reset()





