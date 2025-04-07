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
        
        detect_gate(frame)

def test():
    frame = cv2.imread("/home/tuisku/Pictures/portti2.jpeg", cv2.IMREAD_COLOR)
    detect_gate(frame)


def detect_gate(image):
    frame = image
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
    lower_plywood_bgr0 = np.array([90, 120, 150])
    upper_plywood_bgr0 = np.array([140, 170, 210])
    plywood_mask_bgr0 = cv2.inRange(blur, lower_plywood_bgr0, upper_plywood_bgr0)

    lower_plywood_bgr1 = np.array([130, 160, 190])
    upper_plywood_bgr1 = np.array([160, 190, 230])
    plywood_mask_bgr1 = cv2.inRange(blur, lower_plywood_bgr1, upper_plywood_bgr1)

    lower_plywood_bgr2 = np.array([150, 180, 210])
    upper_plywood_bgr2 = np.array([190, 220, 240])
    plywood_mask_bgr2 = cv2.inRange(blur, lower_plywood_bgr2, upper_plywood_bgr2)

    lower_plywood_bgr3 = np.array([170, 200, 230])
    upper_plywood_bgr3 = np.array([230, 240, 255])
    plywood_mask_bgr3 = cv2.inRange(blur, lower_plywood_bgr3, upper_plywood_bgr3)

    #lower_plywood_bgr4 = np.array([180, 200, 230])
    #upper_plywood_bgr4 = np.array([220, 240, 255])
    #plywood_mask_bgr4 = cv2.inRange(frame, lower_plywood_bgr4, upper_plywood_bgr4)

    # Combine all plywood masks
    plywood_mask_bgr = cv2.bitwise_or(plywood_mask_bgr1, plywood_mask_bgr2)
    plywood_mask_bgr = cv2.bitwise_or(plywood_mask_bgr, plywood_mask_bgr0)
    plywood_mask_bgr = cv2.bitwise_or(plywood_mask_bgr, plywood_mask_bgr3)

    morph = plywood_mask_bgr
    morph = cv2.erode(morph,kernel,iterations = 2)
    morph = cv2.dilate(morph,kernel,iterations = 2)

    plywood_mask_bgr = morph

    # combined mask
    combined_mask = cv2.bitwise_or(mask, plywood_mask_bgr)

    # Display the combined mask
    res = cv2.bitwise_and(frame,frame, mask= combined_mask)
    cv2.imshow("Combined Mask", res)
    cv2.waitKey(1)

    # Display the frame with detected shapes
    #cv2.imshow("Shape Detection", frame)


def main(args=None):
    #while True:
    #    test()
    
    rclpy.init(args=args)
    gate_detector = GateDetector()
    rclpy.spin(gate_detector)
    gate_detector.destroy_node()
    rclpy.shutdown()
    

if __name__ == '__main__':
    main()
