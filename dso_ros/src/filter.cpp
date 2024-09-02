#include <ros/ros.h>
#include <pcl/common/transforms.h>
#include <pcl/common/projection_matrix.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/eigen.hpp>
#include <pcl/io/pcd_io.h>

typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

int main( int argc, char** argv )
{
	ros::init(argc, argv, "filter");
	ros::NodeHandle nh("~");

    PointCloudT::Ptr cloud_show(new PointCloudT());
    pcl::io::loadPCDFile("/home/devil/Documents/test_pcd.pcd", *cloud_show);

    pcl::VoxelGrid<PointT> fil;
    PointCloudT::Ptr tmp(new PointCloudT());
    fil.setLeafSize( 0.05f, 0.05f, 0.05f);
    fil.setInputCloud(cloud_show);
    fil.filter(*tmp);

    pcl::visualization::CloudViewer viewer("simple cloud viewer");
    viewer.showCloud(tmp);
    cout << "show global map, size=" << tmp->points.size() << endl;
    while (!viewer.wasStopped())
    {
        
    }

	return 0;
}
