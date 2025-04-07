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
            '/drone1/image_raw',
            self.listener_callback,
            qos_profile=qos_profile)
        self.br = CvBridge()
    
    def listener_callback(self, data):
        self.get_logger().info('Receiving video frame')
        #frame = self.br.imgmsg_to_cv2(data, 'bgr8')
        frame = cv2.imread("/home/sandra/Downloads/circle_gate.jpeg", cv2.IMREAD_COLOR)
        cv2.imshow("Received frame", frame)
        cv2.waitKey(1)

        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_y = np.array([35, 40, 40])
        upper_y = np.array([80, 255, 255])

        mask = cv2.inRange(frame_hsv, lower_y, upper_y)
        cv2.imshow("Green values", mask)
        cv2.waitKey(1)

        # Preprocess the image with conversion to grayscale and gaussian blur to reduce noice in the image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_with_green_mask = cv2.bitwise_or(gray, mask)

        cv2.imshow("Gray with mask", gray_with_green_mask)
        cv2.waitKey(1)

        blur = cv2.GaussianBlur(gray_with_green_mask, (15, 15), 0)
        cv2.imshow("Blurred", blur)
        cv2.waitKey(1)
        
        # Detect edges, highlighting the significant transitions in pixel intensity (usually correspons to object boundaries)
        # ADJUST THE THRESHOLDS TO CONTROL THE SENSITIVITY OF EDGE DETECTION
        edges = cv2.Canny(blur, 100, 150)

        kernel = np.ones((5,5),np.uint8)
        dilation = cv2.dilate(edges,kernel,iterations = 1)

        # Find countours in the edge-detected image (adjust the second argument for other contour retrieval modes)
        contours, _ = cv2.findContours(dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        ### TEST CODE STARTS ###
        output = frame.copy()
        cv2.drawContours(output, contours, -1, (0,255,0), 3)
        cv2.imshow("Contoured frame", output)
        cv2.waitKey(1)

        selected_contours = []
        biggest_area = 0
        ### TEST CODE ENDS ###

        for contour in contours:
            ### TEST CODE STARTS ###
            area = cv2.contourArea(contour)

            if area > 20000:
                selected_contours.append(contour)
                if area > biggest_area:
                    biggest_area = area
            ### TEST CODE ENDS ###
            
            # Approximating contours: simplify the contour while preserving its core structure
            # Number of vertices is reduced here:
            # Increasing the epsilon value leads to better contour smoothness
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            #cv2.drawContours(frame.copy(), [approx], 0, (0,255,0), 2)

            # Classification of shapes based on the number of vertices found in the approximated contours
            vertices = len(approx)
            if vertices == 4:
                shape = "rectangle"
            else:
                shape = "circle"

            #output = frame.copy()
            #cv2.drawContours(output, [approx], 0, (0,255,0), 2)
            #cv2.imshow("Approximated Contour", output)
            #cv2.waitKey(1)

        ### TEST CODE STARTS ###
        output2 = frame.copy()
        print(len(selected_contours))
        print(biggest_area)
        cv2.drawContours(output2, selected_contours, -1, (255,0,0), 3)
        cv2.imshow("Selected contour frame", output2)
        cv2.waitKey(1)
        ### TEST CODE ENDS###
            

        #cv2.destroyAllWindows()



def main(args=None):
    rclpy.init(args=args)
    gate_detector = GateDetector()
    rclpy.spin(gate_detector)
    gate_detector.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
