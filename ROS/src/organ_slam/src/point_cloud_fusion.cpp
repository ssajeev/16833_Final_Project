

#include <functional>
#include <string>

#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl/sample_consensus/sac_model_sphere.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/transforms.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_listener.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/common/copy_point.h>
#include <pcl/common/io.h>

//#include <point_cloud_fusion.h>

class PointCloudFusion {
  protected:
    // This is primarily to save on typing(!)
    typedef pcl::PointCloud<pcl::PointXYZ> point_cloud_t;

    // The fused point cloud itself
    point_cloud_t fused_cloud;

    // Listener for tf frames
    tf::TransformListener tf_listener;

    // The name of the base frame
    std::string  base_frame_id_cur;

    ros::Publisher pub;

    // publish the fused cloud
    void publish() const {
      // temporary PointCloud2 intermediary
      pcl::PCLPointCloud2 tmp_pc;

      // Convert fused from PCL native type to ROS
      pcl::toPCLPointCloud2(fused_cloud, tmp_pc);
      sensor_msgs::PointCloud2 published_pc;
      pcl_conversions::fromPCL(tmp_pc, published_pc);


      published_pc.header.frame_id = base_frame_id_cur;

      // Publish the data
      pub.publish(published_pc);
    }

  public:
    PointCloudFusion(const std::string& base_frame_id, const ros::Publisher& pub_new) : pub(pub_new) {
      set_base_frame_id(base_frame_id_cur);
    }
    ~PointCloudFusion() { }

    // get the base frame id
    const std::string base_frame_id() const { return base_frame_id_cur; }

    // update base frame id - this will reset the fused point cloud
    void set_base_frame_id(const std::string& new_base_frame_id) {
      // clear current fused point cloud on a base frame change
      fused_cloud.clear();

      // record new frame
      base_frame_id_cur = new_base_frame_id;
    }

    // callback when a new point cloud is available
    void add_cloud(const sensor_msgs::PointCloud2& ros_pc)
    {
      // temporary PointCloud2 intermediary
      pcl::PCLPointCloud2 tmp_pc;

      // transform the point cloud into base_frame_id
      sensor_msgs::PointCloud2 trans_ros_pc;
      //if(!pcl_ros::transformPointCloud(base_frame_id, ros_pc, trans_ros_pc, tf_listener)) {
        // Failed to transform
      //   ROS_WARN("Dropping input point cloud");
      //  return;
      //}
      // Convert ROS point cloud to PCL point cloud
      // See http://wiki.ros.org/hydro/Migration for the source of this magic.
      pcl_conversions::toPCL(trans_ros_pc, tmp_pc);

      pcl::PointCloud<pcl::PointXYZRGB>::Ptr final (new pcl::PointCloud<pcl::PointXYZRGB>);
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (tmp_pc);
      std::vector<int> inliers;

      // created RandomSampleConsensus object and compute the appropriated model
      pcl::SampleConsensusModelSphere<pcl::PointXYZ>::Ptr
        model_s(new pcl::SampleConsensusModelSphere<pcl::PointXYZ> (cloud));

      pcl::RandomSampleConsensus<pcl::PointXYZ> ransac (model_s);
      ransac.setDistanceThreshold (.01);
      ransac.computeModel();
      ransac.getInliers(inliers);

      // copies all inliers of the model computed to another PointCloud
      pcl::copyPointCloud (*cloud, inliers, *final);


      // Convert point cloud to PCL native point cloud
      point_cloud_t input;
      pcl::fromPCLPointCloud2(*final, input);

      // Fuse
      fused_cloud += input;

      // Publish fused cloud
      publish();
    }
};

int main (int argc, char** argv)
{
  // Initialize ROS
  ros::init (argc, argv, "point_cloud_fusion");
  ros::NodeHandle nh;

  // Create a publisher for the fused data and create a PointCloudFusion object to do it.
  PointCloudFusion fusion("base_frame", nh.advertise<sensor_msgs::PointCloud2>("fusion_point", 1));
  // Create a ROS subscriber for the input point cloud
  ros::Subscriber sub = nh.subscribe("point_cloud_smooth", 2, &PointCloudFusion::add_cloud, &fusion);

  // Spin
  ros::spin ();
}
