import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

from rclpy.qos import QoSProfile, QoSReliabilityPolicy

from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO


class ObjectDetectionNode(Node):

    def __init__(self):
        super().__init__('object_detection_node')
        qos_profile = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, depth=1)
        self.subscription = self.create_subscription(Image, '/drone1/image_raw', self.listener_callback, qos_profile=qos_profile)
        self.bridge = CvBridge()

    def listener_callback(self, msg):

        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        print("ok")
        cv2.imshow("camera1", cv_image)
        cv2.waitKey(1)

        yolo = YOLO('yolov8s.pt')

        results = yolo.track(cv_image, stream=True)

        for result in results:
            # get the classes names
            classes_names = result.names

            # iterate over each box
            for box in result.boxes:
                # check if confidence is greater than 40 percent
                if box.conf[0] > 0.4:
                    # get coordinates
                    [x1, y1, x2, y2] = box.xyxy[0]
                    # convert to int
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # get the class
                    cls = int(box.cls[0])

                    # get the class name
                    class_name = classes_names[cls]

                    # get the respective colour
                    colour = (255, 0, 0)

                    # draw the rectangle
                    cv2.rectangle(cv_image, (x1, y1), (x2, y2), colour, 2)

                    # put the class name and confidence on the image
                    cv2.putText(cv_image, f'{classes_names[int(box.cls[0])]} {box.conf[0]:.2f}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2)
                
        # show the image
        cv2.imshow('frame', cv_image)

def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()    

