
// #ifndef SR_NODE_EXAMPLE_CORE_H
// #define SR_NODE_EXAMPLE_CORE_H

// ROS includes.
#include "ros/ros.h"
#include "ros/time.h"

//
// // Dynamic reconfigure includes.
// #include <dynamic_reconfigure/server.h>
// // Auto-generated from cfg/ directory.
// #include <node_example/node_example_paramsConfig.h>

using std::string;

class PointCloudFusion
{
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
public:
  //! Constructor.
  PointCloudFusion();

  //! Destructor.
  ~PointCloudFusion();

  // get the base frame id
  const std::string base_frame_id()

  //! Publish the message.
  void set_base_frame_id(const std::string& new_base_frame_id)

  // callback when a new point cloud is available
  void add_cloud(const sensor_msgs::PointCloud2& ros_pc)



};

#endif // SR_NODE_EXAMPLE_CORE_H
