

#include <functional>
#include <string>

#include <pcl/conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/transforms.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_listener.h>

class PointCloudFusion {
  protected:
    // This is primarily to save on typing(!)
    typedef pcl::PointCloud<pcl::PointXYZ> point_cloud_t;

    // The fused point cloud itself
    point_cloud_t fused_cloud;

    // Listener for tf frames
    tf::TransformListener tf_listener;

    // The name of the base frame
    std::string  base_frame_id;

    ros::Publisher pub;

    // publish the fused cloud
    void publish_() const {
      // temporary PointCloud2 intermediary
      pcl::PCLPointCloud2 tmp_pc;

      // Convert fused from PCL native type to ROS
      pcl::toPCLPointCloud2(fused_cloud, tmp_pc);
      sensor_msgs::PointCloud2 published_pc;
      pcl_conversions::fromPCL(tmp_pc, published_pc);

      published_pc.header.frame_id = base_frame_id;

      // Publish the data
      pub.publish(published_pc);
    }

  public:
    PointCloudFusion(const std::string& base_frame_id, const ros::Publisher& pub) : pub_(pub) {
      set_base_frame_id(base_frame_id);
    }
    ~PointCloudFusion() { }

    // get the base frame id
    const std::string base_frame_id() const { return base_frame_id; }

    // update base frame id - this will reset the fused point cloud
    void set_base_frame_id(const std::string& new_base_frame_id) {
      // clear current fused point cloud on a base frame change
      fused_cloud.clear();

      // record new frame
      base_frame_id = new_base_frame_id;
    }

    // callback when a new point cloud is available
    void add_cloud(const sensor_msgs::PointCloud2& ros_pc)
    {
      // temporary PointCloud2 intermediary
      pcl::PCLPointCloud2 tmp_pc;

      // transform the point cloud into base_frame_id
      sensor_msgs::PointCloud2 trans_ros_pc;
      if(!pcl_ros::transformPointCloud(base_frame_id, ros_pc, trans_ros_pc, tf_listener)) {
        // Failed to transform
        ROS_WARN("Dropping input point cloud");
        return;
      }

      // Convert ROS point cloud to PCL point cloud
      // See http://wiki.ros.org/hydro/Migration for the source of this magic.
      pcl_conversions::toPCL(trans_ros_pc, tmp_pc);

      // Convert point cloud to PCL native point cloud
      point_cloud_t input;
      pcl::fromPCLPointCloud2(tmp_pc, input);

      // Fuse
      fused_cloud += input;

      // Implemment icp
      pcl::IterativeCLosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
      icp.setInputSource(input)
      icp.setInputTarget(fused_cloud)
      pcl::PointCloud<pcl::PointXYZ> Final;

      // Publish fused cloud
      publish();
    }
};

int main (int argc, char** argv)
{
  // Initialize ROS
  ros::init (argc, argv, "pcl_fuse");
  ros::NodeHandle nh;

  // Create a publisher for the fused data and create a PointCloudFusion object to do it.
  PointCloudFusion fusion("/odom", nh.advertise<sensor_msgs::PointCloud2>("/fused_points", 1));

  // Create a ROS subscriber for the input point cloud
  ros::Subscriber sub = nh.subscribe("/camera/depth/points", 1, &PointCloudFusion::add_cloud, &fusion);

  // Spin
  ros::spin ();
}
