import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, QoSReliabilityPolicy

import cv2
import numpy as np
from tello_msgs.srv import TelloAction

import torch

import time
from djitellopy import Tello # test

class GateDetector(Node):
    def __init__(self):
        super().__init__('gate_detector')
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            depth=1
        )
        self.processing_image = False # flag to ignore incoming images while processing the current one
        self.publication = self.create_publisher(String, '/cmd_string', 10)
        self.client = self.create_client(TelloAction, '/tello_action')
        self.publisher = self.create_publisher(Image, '/larggest_gate', 10)
        self.subscription = self.create_subscription(
            Image,
            '/image_raw',
            self.listener_callback,
            10) # qos_profile=qos_profile
        self.br = CvBridge()
        self.model = torch.hub.load('../../../../Documents/ultralytics', 'custom', source='local', path='../../../../Documents/ultralytics/weights/bests640e75n1010.pt', force_reload=True)
        self.model.eval()
        self.num_of_gates = 0
        self.flying = False
        print("model inited...")

    def service_response_callback(self, future):
        try:
            response = future.result()
            print(f"Service call succeeded: {response}")
        except Exception as e:
            print(f"Service call failed: {e}")

    # Listener callback/main thing
    def listener_callback(self, data):
        self.get_logger().info('Receiving video frame')
        if self.processing_image:
            return
        self.processing_image = True

        self.get_logger().info('Receiving video frame')
        frame = self.br.imgmsg_to_cv2(data, 'bgr8')

        if self.detect_takeoff_signal(frame):
            while not self.client.wait_for_service(timeout_sec=1.0):
                print('Service not available, waiting...')
            request = TelloAction.Request()
            self.get_logger().info('Takeoff command')
            if not self.flying:
                request.cmd = 'takeoff'
                self.flying = True
            else:
                request.cmd = 'land'
                self.flying = False
            future = self.client.call_async(request)
            future.add_done_callback(self.service_response_callback)
        if self.num_of_gates:
            self.find_stop_signal(frame)
        if self.num_of_gates == 4 and self.detect_stop_signal(frame):
            while not self.client.wait_for_service(timeout_sec=1.0):
                print('Service not available, waiting...')

            request = TelloAction.Request()
            self.get_logger().info('Land command')
            request.cmd = 'land'

            future = self.client.call_async(request)
            future.add_done_callback(self.service_response_callback)

        self.detect_gate(frame)

        self.processing_image = False

    # Create a mask for the specified color (red for stop signal and yellow for start signal)
    # Green is not in use, we are using Lipei's model instead
    def create_mask(self, image, color):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        if color == 'red':
            lower_red1 = np.array([0, 50, 50])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 150, 50])
            upper_red2 = np.array([180, 255, 255])

            mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
            red_mask = cv2.bitwise_or(mask_red1, mask_red2)

            return red_mask

        elif color == 'green':
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower_green = np.array([60, 80, 20])
            upper_green = np.array([85, 255, 240])
            mask = cv2.inRange(hsv, lower_green, upper_green)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
            morph = mask
            morph = cv2.erode(morph,kernel,iterations = 2)
            morph = cv2.dilate(morph,kernel,iterations = 3)

            green_mask = morph
            return green_mask

        elif color == 'yellow':
            lower_yellow = np.array([10, 80, 80])
            upper_yellow = np.array([50, 255, 255])
            yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

            return yellow_mask

    # Check if the midpoint is aligned vertically, if not move up or down
    def check_up_down(self, error_height, height_threshold):
        if abs(error_height) > height_threshold:
            print("send up/down")
            aligned = False
            msg = String()
            if error_height > 0:
                msg.data = 'down'
            else:
                msg.data = 'up'
            self.publication.publish(msg)
            time.sleep(1.5)

    # Check if the midpoint is aligned horizontally, if not move up or down
    def check_right_left(self, error_width, width_threshold):
        if abs(error_width) > width_threshold:
            print("send right/left")
            self.aligned = False
            msg = String()
            if error_width > 0:
                msg.data = "right"
            else:
                msg.data = "left"
            self.publication.publish(msg)
            time.sleep(1.5)

    # Check for the stop sign, and navigate towards it
    def find_stop_signal(self, image):
        frame = image

        image_height = frame.shape[0]
        image_width = frame.shape[1]

        # Image centerpoint, a bit higher than the middle of the image because the camera points down
        image_center_height = image_height / 3
        image_center_width = image_width / 2

        # How large area is considered to be midpoint good enough
        height_threshold = image_height / 9
        width_threshold = image_width / 9

        red_mask = self.create_mask(frame, 'red')
        edges = cv2.Canny(red_mask, 50, 150)
        # Find contours in the edge-detected image
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Calculate the center of the red area
        moments = cv2.moments(red_mask)
        if moments["m00"] != 0:
            center_x = int(moments["m10"] / moments["m00"])
            center_y = int(moments["m01"] / moments["m00"])
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
            cv2.putText(frame, f"Center: ({center_x}, {center_y})", (center_x + 10, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        else:
            center_x, center_y = None, None

        # If the center of the red area is found, check if it is aligned
        if center_x is not None and center_y is not None:
            aligned = True
            # Check error in height and width
            error_height = center_y - image_center_height
            error_width = center_x - image_center_width
            # Check horizontal and vertical misalignment, the functions handle the movement if necessary
            if abs(error_height) > height_threshold:
                aligned = False
                check_up_down(error_height, height_threshold)
            if abs(error_width) > width_threshold:
                aligned = False
                check_right_left(error_width, width_threshold)

            # If the red area is aligned, move forward
            if aligned:
                print("aligned with red area")
                msg = String()
                msg.data = "forward"
                self.publication.publish(msg)
                time.sleep(1.5) # stabilization time

    # Check for the takeoff signal (mostly yellow screen). If found, send a takeoff command
    def detect_takeoff_signal(self, image):
        frame = image
        blur = cv2.GaussianBlur(frame,(15,15),0)
        cv2.waitKey(1)

        yellow_mask = self.create_mask(frame, 'yellow')

        # Apply the yellow mask to the image
        yellow_result = cv2.bitwise_and(frame, frame, mask=yellow_mask)

        # Check if more than 75% of the image is yellow
        white_pixels = cv2.countNonZero(yellow_mask)
        total_pixels = yellow_mask.shape[0] * yellow_mask.shape[1]
        yellow_ratio = white_pixels / total_pixels

        if yellow_ratio > 0.50:
            return 1
        return 0

    # Detect the largest gate in the image
    # The function uses the YOLOv5 model to detect the gates
    def detect_gate(self, image):
        frame = image
        image_height = frame.shape[0]
        image_width = frame.shape[1]

        # Image centerpoint, a bit higher than the middle of the image because the camera points down
        image_center_height = image_height / 3
        image_center_width = image_width / 2

        # How large area is considered to be midpoint good enough
        height_threshold = image_height / 9
        width_threshold = image_width / 9
        
        results = self.model(frame)
        #results = []
        detections = results.xyxy[0]

        #draw result
        for *box, conf, cls in detections:
            x1, y1, x2, y2 = map(int, box)
            label = self.model.names[int(cls)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    
        largest_gate = None
        largest_area = 0

        # Loop through detections and find the largest gate
        for *box, conf, cls in detections:
            label = self.model.names[int(cls)]

            x1, y1, x2, y2 = map(int, box)
            width = x2 - x1
            height = y2 - y1
            area = width * height
            # Check if this gate is larger than the previous largest gate
            if area > largest_area:
                largest_area = area
                largest_gate = (x1, y1, x2, y2, largest_area)

        if largest_gate and self.num_of_gates<4:
            x1, y1, x2, y2, area = largest_gate
            # Draw a red dot in the center of the largest gate
            center_x_gate = (x1 + x2) // 2
            center_y_gate = (y1 + y2) // 2
            cv2.circle(frame, (center_x_gate, center_y_gate), 5, (0, 0, 255), -1)  #
            # Draw larggest gate bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            self.publisher.publish(self.br.cv2_to_imgmsg(frame))

        # If a gate is detected, align the drone with it
        if largest_gate:
            # Align the gate

            # Check error in height and width
            error_height = center_y_gate - image_center_height
            error_width = center_x_gate - image_center_width
            
            #if num_of_gates>3:
            #	height_threshold = image_height / 6
            #	width_threshold = image_width / 6

            # Align the height
            aligned = True
            if abs(error_height) > height_threshold:
                aligned = False
                msg = String()
                if error_height > 0:
                    msg.data = 'down'
                else:
                    msg.data = 'up'
                self.publication.publish(msg)
                time.sleep(1.5) # stabilization time
        
            # Align width
            if abs(error_width) > width_threshold:
                aligned = False
                msg = String()
                if error_width > 0:
                    msg.data = "right"
                else:
                   msg.data = "left"
                self.publication.publish(msg)
                time.sleep(1.5) # stabilization time

            # Move forward if alignment is ok
            if aligned == True:
                msg = String()
                # If the radius of the gate is large enough, move a lot forward
                if self.num_of_gates<4:
                    if (x2-x1) > 0.9*image_height:
                        msg.data = "forwardlong"
                        self.publication.publish(msg)
                        time.sleep(2.5) # stabilization time
                        self.num_of_gates = self.num_of_gates + 1
                        print(f"GATES PASSED {num_of_gates}")
                    # Otherwise take only a small step
                    else:
                        msg.data = "forward"
                        self.publication.publish(msg)
                        time.sleep(1.5) # stabilization time
        else:
            if num_of_gates<4:
                #### UPDATE THE ROTATION DIRECTION BASED ON THE RACING DAY GATE ARRANGEMENT!!!
                msg = String()
                msg.data = "rightsmall"
                self.publication.publish(msg)
                time.sleep(1.5) # stabilization time

    def detect_stop_signal(self, image):
        frame = image
        blur = cv2.GaussianBlur(frame,(15,15),0)
        cv2.waitKey(1)

        # red mask
        red_mask = create_red_mask(frame, 'red')

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
                return 1
        return 0

def test():
    frame = cv2.imread("/home/tuisku/Pictures/portti6.jpg", cv2.IMREAD_COLOR)
    #detect_gate(frame)

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
