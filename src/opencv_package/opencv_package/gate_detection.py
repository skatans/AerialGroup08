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
        
        cv2.waitKey(1)
        # Convert the image to grayscale for shape detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply GaussianBlur to reduce noise and improve edge detection
        frame = cv2.GaussianBlur(frame,(15,15),0)

        '''FILTERING'''
        # green mask
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_green = np.array([40, 20, 20])
        upper_green = np.array([80, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # plywood mask
        lower_plywood_bgr = np.array([150, 150, 150])
        upper_plywood_bgr = np.array([200, 200, 200])
        plywood_mask_bgr = cv2.inRange(frame, lower_plywood_bgr, upper_plywood_bgr)

        # combined mask
        combined_mask = cv2.bitwise_or(mask, plywood_mask_bgr)

        kernel = np.ones((5,5),np.uint8)
        #dilated = cv2.dilate(combined_mask,kernel,iterations = 1)

        # Detect edges using Canny edge detection
        edges = cv2.Canny(combined_mask, 50, 150)

        # Find contours in the edge-detected image
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #frame = combined_mask
        res = cv2.bitwise_and(frame,frame, mask= combined_mask)

        for contour in contours:
            # Approximate the contour to reduce the number of points
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Draw the contour and label the shape
            cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(approx)
            #cv2.putText(frame, shape, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Display the frame with detected shapes
        cv2.imshow("Shape Detection", res)

def test():

    frame = cv2.imread("/home/tuisku/Pictures/portti2.jpeg", cv2.IMREAD_COLOR)
    blur = cv2.GaussianBlur(frame,(15,15),0)
    #cv2.imshow("Received frame", frame)
    cv2.waitKey(1)
    # Convert the image to grayscale for shape detection
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    '''FILTERING'''
    # green mask
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 20, 20])
    upper_green = np.array([80, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((5,5),np.uint8)
    morph = mask
    morph = cv2.erode(morph,kernel,iterations = 2)
    morph = cv2.dilate(morph,kernel,iterations = 2)

    mask = morph

    # plywood mask
    lower_plywood_bgr = np.array([150, 150, 150])
    upper_plywood_bgr = np.array([200, 200, 200])
    plywood_mask_bgr = cv2.inRange(frame, lower_plywood_bgr, upper_plywood_bgr)

    # combined mask
    combined_mask = cv2.bitwise_or(mask, plywood_mask_bgr)

    kernel = np.ones((5,5),np.uint8)
    #pening = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    opening = cv2.dilate(combined_mask,kernel,iterations = 1)

    # Display the combined mask
    res = cv2.bitwise_and(frame,frame, mask= mask)
    cv2.imshow("Combined Mask", res)
    cv2.waitKey(1)

    # Display the frame with detected shapes
    #cv2.imshow("Shape Detection", frame)


def main(args=None):
    while True:
        test()
    
    rclpy.init(args=args)
    gate_detector = GateDetector()
    rclpy.spin(gate_detector)
    gate_detector.destroy_node()
    rclpy.shutdown()
    

if __name__ == '__main__':
    main()
