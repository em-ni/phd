import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class URScriptPublisher(Node):

    def __init__(self):
        super().__init__('urscript_publisher')
        self.publisher_ = self.create_publisher(String, '/urscript_interface/script_command', 10)
        self.timer = self.create_timer(1.0, self.publish_urscript_command) # Timer to publish once after 1 second
        self.get_logger().info('URScript Publisher Node has been started.')
        self.published = False

    def publish_urscript_command(self):
        if not self.published:
            # Define your joint angles in degrees
            joint1_deg = -46.31
            joint2_deg = -114.28
            joint3_deg = -31.57
            joint4_deg = -101.55
            joint5_deg = 44.97
            joint6_deg = -41.10

            msg = String()
            msg.data = f"""def my_prog():

  set_digital_out(1, True)

  # Use d2r() to convert degrees to radians directly in the script
  movej([d2r({joint1_deg}), d2r({joint2_deg}), d2r({joint3_deg}), d2r({joint4_deg}), d2r({joint5_deg}), d2r({joint6_deg})], a=1.0, v=0.20, r=0)

  textmsg(\"motion finished\")

end"""
            self.publisher_.publish(msg)
            self.get_logger().info(f'Publishing: "{msg.data}"')
            self.published = True
            # Optional: Destroy the node after publishing once to mimic --once behavior strictly
            # self.destroy_node()

def main(args=None):
    rclpy.init(args=args)
    urscript_publisher = URScriptPublisher()
    rclpy.spin_once(urscript_publisher) # Spin once to allow the timer to trigger and publish
    urscript_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()