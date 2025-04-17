import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, QoSReliabilityPolicy

import cv2
import numpy as np
from tello_msgs.srv import TelloAction

flying = False

plywood_light = [(10, 30), (0,40), (220,240)]
plywood_mid = [(10, 30), (10,100), (200,240)]
plywood_dark= [(5, 30), (20,100), (130,240)]

class GateDetector(Node):
    def __init__(self):
        super().__init__('gate_detector')
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            depth=1
        )
        self.client = self.create_client(TelloAction, 'tello_action')
        self.subscription = self.create_subscription(
            Image,
            '/image_raw',
            self.listener_callback,
            qos_profile=qos_profile)
        self.br = CvBridge()

    def service_response_callback(self, future):
        try:
            response = future.result()
            print(f"Service call succeeded: {response}")
        except Exception as e:
            print(f"Service call failed: {e}")

    def listener_callback(self, data):
        #self.get_logger().info('Receiving video frame')
        frame = self.br.imgmsg_to_cv2(data, 'bgr8')

        area = detect_stop(frame)
        #print(area)
        if area > 0:
            while not self.client.wait_for_service(timeout_sec=1.0):
                print('Service not available, waiting...')

            request = TelloAction.Request()
            if area == 1:
                self.get_logger().info('Takeoff command')
                request.cmd = 'takeoff'
            elif area > 5000:
                self.get_logger().info('Land command')
                request.cmd = 'land'
            else:
                self.get_logger().info('Land command')
                request.cmd = 'land'

            future = self.client.call_async(request)

            future.add_done_callback(self.service_response_callback)

        detect_gate(frame)

def test():
    frame = cv2.imread("/home/tuisku/Pictures/portti6.jpg", cv2.IMREAD_COLOR)
    detect_gate(frame)

def detect_stop(image):
    frame = image
    blur = cv2.GaussianBlur(frame,(15,15),0)
    cv2.waitKey(1)

    # yellow mask
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([10, 80, 80])
    upper_yellow = np.array([50, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Apply the yellow mask to the image
    yellow_result = cv2.bitwise_and(frame, frame, mask=yellow_mask)

    # Check if more than 75% of the image is yellow
    white_pixels = cv2.countNonZero(yellow_mask)
    total_pixels = yellow_mask.shape[0] * yellow_mask.shape[1]
    white_ratio = white_pixels / total_pixels

    if white_ratio > 0.75:
        return 1

    # red mask
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 150, 50])
    upper_red2 = np.array([180, 255, 255])

    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask_red2

    # Apply the red mask to the image
    red_result = cv2.bitwise_and(frame, frame, mask=red_mask)

    red_edges = cv2.Canny(red_result, 50, 150)
    # Find contours in the edge-detected image
    red_contours, _ = cv2.findContours(red_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Check for a big red area in the contours
    for contour in red_contours:
        area = cv2.contourArea(contour)
        cv2.putText(red_result, f"Area: {area}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        if area > 3000:  # Threshold for a "big" red area
            #print(area)
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Big Red Area", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow("Red Mask Applied", red_result)
            cv2.waitKey(1)
            return area
    return 0


def detect_gate(image):
    frame = image
    blur = cv2.GaussianBlur(frame,(15,15),0)
    cv2.waitKey(1)

    '''FILTERING'''
    # green mask
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([60, 80, 20])
    upper_green = np.array([85, 255, 240])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    morph = mask
    morph = cv2.erode(morph,kernel,iterations = 2)
    morph = cv2.dilate(morph,kernel,iterations = 3)

    mask = morph
    
    # Check the lightness of the image
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lightness = np.mean(hsv[:, :, 2])  # V channel represents lightness
    #print("Lightness:", lightness)

    # Choose the plywood array based on lightness
    if lightness > 180:
        plywood = plywood_light
    elif lightness > 120:
        plywood = plywood_mid
    else:
        plywood = plywood_dark

    # plywood mask using HSV
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    lower_plywood_hsv = np.array([plywood[0][0], plywood[1][0], plywood[2][0]])
    upper_plywood_hsv = np.array([plywood[0][1], plywood[1][1], plywood[2][1]])
    plywood_mask_hsv = cv2.inRange(hsv, lower_plywood_hsv, upper_plywood_hsv)

    morph = plywood_mask_hsv
    morph = cv2.erode(morph, kernel, iterations=1)
    morph = cv2.dilate(morph, kernel, iterations=3)

    plywood_mask_hsv = morph

    # white mask
    lower_white = np.array([120, 0, 150])
    upper_white = np.array([160, 40, 200])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    morph = mask_white
    morph = cv2.erode(morph,kernel,iterations = 1)
    morph = cv2.dilate(morph,kernel,iterations = 10)
    morph = cv2.erode(morph,kernel,iterations = 5)
    mask_white = morph

    # combined mask
    combined_mask = cv2.bitwise_or(mask, plywood_mask_hsv)
    #combined_mask = cv2.bitwise_or(combined_mask, mask_white)
    morph = cv2.dilate(combined_mask, kernel, iterations=2)
    combined_mask = morph

    # Display the combined mask
    res = cv2.bitwise_and(frame,frame, mask= combined_mask)
    #cv2.imshow("Combined Mask", combined_mask)
    #cv2.waitKey(1)

    res = combined_mask
    # Detect edges using Canny edge detection
    edges = cv2.Canny(res, 50, 150)
    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest circle and midpoints of all circles
    largest_circle = None
    largest_radius = 0
    circle_midpoints = []

    # Draw contours on the original frame
    for contour in contours:
        continue
        area = cv2.contourArea(contour)
        #if area < 3000:
        #    continue

        # Fit a minimum enclosing circle around the contour
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)

        if radius < 50:
            continue

        # Store the midpoint and dimensions of the circle
        circle_midpoints.append((center, radius))

        # Check if this is the largest circle
        if radius > largest_radius:
            largest_radius = radius
            largest_circle = (center, radius)


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

        # Get the width and height of the bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        width = w
        height = h
        # Calculate the aspect ratio
        aspect_ratio = float(width) / height
        print("Aspect Ratio:", aspect_ratio)
        # Check if the aspect ratio is within a certain range
        if aspect_ratio < 0.5 or aspect_ratio > 2:
            continue
        elif aspect_ratio < 0.9:
            cv2.putText(frame, "danger", (x, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        
        # Draw the contour and label the shape
        cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
        x, y, w, h = cv2.boundingRect(approx)
        cv2.putText(frame, shape, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        # Draw the bounding rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        '''
        (x,y),radius = cv2.minEnclosingCircle(contour)
        center = (int(x),int(y))
        radius = int(radius)
        cv2.circle(frame,center,radius,(0,255,0),2)
        ellipse = cv2.fitEllipse(contour)
        cv2.ellipse(frame,ellipse,(0,255,0),2)
        '''

    # Draw the largest circle
    if largest_circle:
        center, radius = largest_circle
        cv2.circle(frame, center, radius, (0, 255, 255), 2)
        cv2.putText(frame, f"Radius: {radius}", (center[0] - 40, center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Draw all circle midpoints
    for center, radius in circle_midpoints:
        cv2.circle(frame, center, 3, (255, 0, 0), -1)
        cv2.putText(frame, f"({center[0]}, {center[1]})", (center[0] + 5, center[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    #frame = res
    #cv2.imshow("Shape Detection", frame)


def main(args=None):
    while False:
        test()
    
    rclpy.init(args=args)
    gate_detector = GateDetector()
    rclpy.spin(gate_detector)
    gate_detector.destroy_node()
    rclpy.shutdown()
    

if __name__ == '__main__':
    main()
