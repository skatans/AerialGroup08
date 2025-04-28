import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from geometry_msgs.msg import Twist

import time

class CommandSubscriber(Node):
    def __init__(self):
        super().__init__('cmd_subscriber')
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.subscription = self.create_subscription(
            String,
            '/cmd_string',
            self.listener_callback,
            10)
        self.processing = False  # Flag to prevent processing new data
    
    def listener_callback(self, msg):
        if self.processing:
            self.get_logger().info('Ignoring new data during delay')
            return
        self.get_logger().info('Receiving command')
        pub_msg = Twist()
        pub_msg.linear.x = 0.0
        pub_msg.linear.y = 0.0
        pub_msg.linear.z = 0.0
        pub_msg.angular.x = 0.0
        pub_msg.angular.y = 0.0
        pub_msg.angular.z = 0.0

        if (msg.data == "left"):
            self.get_logger().info('Turning left')
            pub_msg.angular.z = 0.25
            self.publisher.publish(pub_msg)
        elif (msg.data == "right"):
            self.get_logger().info('Turning right')
            pub_msg.angular.z = -0.25
            self.publisher.publish(pub_msg)
        elif (msg.data == "forward"):
            self.get_logger().info('Going forward')
            pub_msg.linear.x = 0.15
            self.publisher.publish(pub_msg)
        elif (msg.data == "forwardlong"):
            self.get_logger().info('Going forward long')
            self.processing = True
            pub_msg.linear.z = -0.15
            self.publisher.publish(pub_msg)
            time.sleep(1.0)
            pub_msg.linear.x = 0.4
            self.publisher.publish(pub_msg)
            time.sleep(2.0)
            pub_msg.linear.z = 0.15
            self.publisher.publish(pub_msg)
            self.processing = False
        elif (msg.data == "back"):
            self.get_logger().info('Going back')
            pub_msg.linear.x = -0.15
            self.publisher.publish(pub_msg)
        elif (msg.data == "up"):
            self.get_logger().info('Going up')
            pub_msg.linear.z = 0.15
            self.publisher.publish(pub_msg)
        elif (msg.data == "down"):
            self.get_logger().info('Going down')
            pub_msg.linear.z = -0.15
            self.publisher.publish(pub_msg)
        else:
            self.get_logger().info('Dont know what to do')

        time.sleep(1.0)
        pub_msg.linear.x = 0.0
        pub_msg.linear.y = 0.0
        pub_msg.linear.z = 0.0
        pub_msg.angular.x = 0.0
        pub_msg.angular.y = 0.0
        pub_msg.angular.z = 0.0
        self.publisher.publish(pub_msg)

def main(args=None):
    rclpy.init(args=args)
    cmd_subscriber = CommandSubscriber()
    rclpy.spin(cmd_subscriber)
    cmd_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()