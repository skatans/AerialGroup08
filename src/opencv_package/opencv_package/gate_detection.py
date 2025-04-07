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
        # Convert the image to grayscale for shape detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply GaussianBlur to reduce noise and improve edge detection
        blur = cv2.GaussianBlur(gray,(15,15),0)
        #th3 = cv2.medianBlur(gray,5)
        #th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
        #ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        '''
        # Rectangular Kernel
        kernel_rq = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
        # Elliptical Kernel
        kernel_el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        sharpen = cv2.filter2D(blur, -1, kernel_rq)
        '''

        # Detect edges using Canny edge detection
        edges = cv2.Canny(blur, 50, 150)

        # Find contours in the edge-detected image
        contours, _ = cv2.findContours(blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        frame = blur

        for contour in contours:
            # Approximate the contour to reduce the number of points
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Check the number of vertices in the approximated contour
            vertices = len(approx)

            if vertices == 4:
                shape = "Square/Rectangle"
            elif vertices > 4:
                shape = "Circle"
            else:
                shape = "Other"

            # Draw the contour and label the shape
            cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(approx)
            cv2.putText(frame, shape, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Display the frame with detected shapes
        cv2.imshow("Shape Detection", frame)

def test():

    frame = cv2.imread("/home/tuisku/Pictures/portti.jpeg", cv2.IMREAD_COLOR)
    frame = cv2.GaussianBlur(frame,(25,25),0)
    #cv2.imshow("Received frame", frame)
    cv2.waitKey(1)
    # Convert the image to grayscale for shape detection
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # mask
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    #cv2.imshow("Green values", mask)

    # Define the range for plywood in BGR
    lower_plywood_bgr = np.array([200, 200, 200])
    upper_plywood_bgr = np.array([250, 250, 250])
    plywood_mask_bgr = cv2.inRange(frame, lower_plywood_bgr, upper_plywood_bgr)

    # Display the mask for plywood
    cv2.imshow("Plywood mask (BGR)", plywood_mask_bgr)
    cv2.waitKey(1)

    # Combine the whites from both masks
    combined_mask = cv2.bitwise_or(mask, plywood_mask_bgr)

    kernel = np.ones((5,5),np.uint8)
    #pening = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    opening = cv2.dilate(combined_mask,kernel,iterations = 1)

    # Display the combined mask
    cv2.imshow("Combined Mask", opening)
    cv2.waitKey(1)

    # Display the frame with detected shapes
    #cv2.imshow("Shape Detection", frame)


def main(args=None):
    while True:
        test()
    '''
    rclpy.init(args=args)
    gate_detector = GateDetector()
    rclpy.spin(gate_detector)
    gate_detector.destroy_node()
    rclpy.shutdown()
    '''

if __name__ == '__main__':
    main()
