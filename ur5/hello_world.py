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
            msg = String()
            msg.data = 'popup("hello from Python")'
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