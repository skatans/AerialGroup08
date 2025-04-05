import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, QoSReliabilityPolicy

import cv2
import numpy as np

class GateDetector(Node):
    def __init__(self):
        super().__init__('gate_detector')
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            depth=1
        )
        self.subscription = self.create_subscription(
            Image,
            '/image_raw',
            self.listener_callback,
            qos_profile=qos_profile)
        self.br = CvBridge()
    
    def listener_callback(self, data):
        self.get_logger().info('Receiving video frame')
        frame = self.br.imgmsg_to_cv2(data, 'bgr8')
        #frame = cv2.imread("/home/sandra/Downloads/circle_gate.jpeg", cv2.IMREAD_COLOR)
        #cv2.imshow("Received frame", frame)
        cv2.waitKey(1)

        # Preprocess the image with conversion to grayscale and gaussian blur to reduce noice in the image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (15, 15), 0)

        ### TEST CODE STARTS ###
        #kernel_25 = np.ones((25,25), dtype=np.float32) / 625.0

        #output_kernel = cv2.filter2D(gray, -1, kernel_25)
        #blur = cv2.blur(gray, (25,25))

        #cv2.imshow("Blurred frame", blur)
        cv2.waitKey(1)
        ### TEST CODE ENDS ##

        # Detect edges, highlighting the significant transitions in pixel intensity (usually correspons to object boundaries)
        # ADJUST THE THRESHOLDS TO CONTROL THE SENSITIVITY OF EDGE DETECTION
        edges = cv2.Canny(blur, 30, 100)

        # Find countours in the edge-detected image (adjust the second argument for other contour retrieval modes)
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        ### TEST CODE STARTS ###
        output = frame.copy()
        cv2.drawContours(output, contours, -1, (0,255,0), 3)
        cv2.imshow("Contoured frame", output)
        cv2.waitKey(1)
        ### TEST CODE ENDS ###

        for contour in contours:
            # Approximating contours: simplify the contour while preserving its core structure
            # Number of vertices is reduced here:
            # Increasing the epsilon value leads to better contour smoothness
            epsilon = 0.04 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            #cv2.drawContours(frame.copy(), [approx], 0, (0,255,0), 2)

            # Classification of shapes based on the number of vertices found in the approximated contours
            vertices = len(approx)
            print(vertices)
            if vertices == 4:
                shape = "rectangle"
            else:
                shape = "circle"

            output = frame.copy()
            cv2.drawContours(output, [approx], 0, (0,255,0), 2)
            #cv2.imshow("Approximated Contour", output)
            cv2.waitKey(1)
            

        #cv2.destroyAllWindows()



def main(args=None):
    rclpy.init(args=args)
    gate_detector = GateDetector()
    rclpy.spin(gate_detector)
    gate_detector.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
