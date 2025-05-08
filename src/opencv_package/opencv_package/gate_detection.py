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
#from djitellopy import Tello # test


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

    # Listener callback/main loop
    def listener_callback(self, data):
        self.get_logger().info('Receiving video frame')
        if self.processing_image:
            return
        self.processing_image = True

        self.get_logger().info('Receiving video frame')
        frame = self.br.imgmsg_to_cv2(data, 'bgr8')

        if not self.flying and self.num_of_gates == 0:
            request = TelloAction.Request()
            self.get_logger().info('Takeoff command')
            request.cmd = 'takeoff'
            self.flying = True
            self.num_of_gates = 4
            future = self.client.call_async(request)
            future.add_done_callback(self.service_response_callback)

            time.sleep(1.5)

            msg = String()
            self.get_logger().info('Sending up command')
            msg.data = 'up'
            self.publication.publish(msg)
            time.sleep(1.5)
        if self.num_of_gates == 4:
            if self.detect_stop_signal(frame):
                while not self.client.wait_for_service(timeout_sec=1.0):
                    print('Service not available, waiting...')
                request = TelloAction.Request()
                self.get_logger().info('Land command')
                request.cmd = 'land'
                #self.flying = False # if uncommented, the drone will go up again afret landing
                future = self.client.call_async(request)
                future.add_done_callback(self.service_response_callback)
            else:
                self.find_stop_signal(frame)

        if self.num_of_gates < 4:
            self.detect_gate(frame)

        self.processing_image = False
    # Create a mask for the specified color (red for stop signal and yellow for start signal)
    # Green is not in use, we are using Lipei's model instead
    def create_mask(self, image, color):
        frame = image
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        if color == 'red':
            lower_red1 = np.array([0, 70, 50])
            upper_red1 = np.array([10, 255, 180])
            lower_red2 = np.array([170, 70, 50])
            upper_red2 = np.array([180, 255, 180])

            mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
            red_mask = cv2.bitwise_or(mask_red1, mask_red2)

            morph = mask_red2
            #morph = cv2.erode(morph,kernel,iterations = 3)
            #morph = cv2.dilate(morph,kernel,iterations = 5)

            return morph

        elif color == 'green':
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower_green = np.array([60, 80, 20])
            upper_green = np.array([85, 255, 240])
            mask = cv2.inRange(hsv, lower_green, upper_green)

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
    
    '''
    Methods to check alignment and send asjustment commands
    '''
    # Check if the midpoint is aligned vertically, if not move up or down
    def check_up_down(self, error_height, height_threshold):
        if abs(error_height) > height_threshold:
            aligned = False
            msg = String()
            if error_height > 0:
                self.get_logger().info('Sending down command, error height: ' + str(error_height))
                msg.data = 'down'
            else:
                self.get_logger().info('Sending up command, error height: ' + str(error_height))
                msg.data = 'up'
            self.publication.publish(msg)
            time.sleep(1.5)

    # Check if the midpoint is aligned horizontally, if not move up or down
    def check_right_left(self, error_width, width_threshold):
        if abs(error_width) > width_threshold:
            self.aligned = False
            msg = String()
            if error_width > 0:
                self.get_logger().info('Sending right command, error width: ' + str(error_width))
                msg.data = "right"
            else:
                self.get_logger().info('Sending left command, error width: ' + str(error_width))
                msg.data = "left"
            self.publication.publish(msg)
            time.sleep(1.5)

    '''
    Methods to find the stop signal and stop the drone
    '''

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
                self.check_up_down(error_height, height_threshold)
            if abs(error_width) > width_threshold:
                aligned = False
                self.check_right_left(error_width, width_threshold)

            # If the red area is aligned, move forward
            if aligned:
                print("aligned with red area")
                msg = String()
                msg.data = "forward"
                self.publication.publish(msg)
                time.sleep(1.5) # stabilization time

    def detect_stop_signal(self, image):
        frame = image
        blur = cv2.GaussianBlur(frame,(15,15),0)
        cv2.waitKey(1)

        # red mask
        red_mask = self.create_mask(frame, 'red')

        # Apply the red mask to the image
        red_result = cv2.bitwise_and(frame, frame, mask=red_mask)

        #cv2.imshow("Red Mask Applied", red_result)
        #cv2.waitKey(1)

        # Define the center region of the frame
        center_height_start = red_result.shape[0] // 3
        center_height_end = 2 * red_result.shape[0] // 3
        center_width_start = red_result.shape[1] // 3
        center_width_end = 2 * red_result.shape[1] // 3

        # Extract the center region
        center_region = red_mask[center_height_start:center_height_end, center_width_start:center_width_end]

        # Count the white pixels in the center region
        white_pixels = cv2.countNonZero(center_region)
        #total_pixels = center_region.shape[0] * center_region.shape[1]
        #white_pixels = cv2.countNonZero(red_mask)
        total_pixels = red_result.shape[0] * red_result.shape[1]
        red_ratio = white_pixels / total_pixels
        #self.get_logger().info('Red ratio :' + str(red_ratio))

        if red_ratio > 0.03:
            return 1
        return 0
    def detect_gate(self, image):
        frame = image
        image_height = frame.shape[0]
        image_width = frame.shape[1]
        
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

        if largest_gate:
            x1, y1, x2, y2, area = largest_gate
            # Draw a red dot in the center of the largest gate
            center_x_gate = (x1 + x2) // 2
            center_y_gate = (y1 + y2) // 2
            cv2.circle(frame, (center_x_gate, center_y_gate), 5, (0, 0, 255), -1)  #
            # Draw larggest gate bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            self.publisher.publish(self.br.cv2_to_imgmsg(frame))
        
        if largest_gate:
            # Align the gate
            image_center_height = image_height / 3
            image_center_width = image_width / 2

            error_height = center_y_gate - image_center_height
            error_width = center_x_gate - image_center_width

            height_threshold = image_height / 10    # 9
            width_threshold = image_width / 10      # 9

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
            self.get_logger().info('Aligned with gate')
            # Move forward if alignment is ok
            if aligned == True:
                msg = String()
                # If the radius of the gate is large enough, move a lot forward
                print(f">>>>>{x2-x1}    {0.9*image_height}")
                if (x2-x1) > 0.9*image_height:
                    msg.data = "forwardlong"
                    self.publication.publish(msg)
                    time.sleep(2.5) # stabilization time
                    self.num_of_gates = self.num_of_gates + 1
                    print(f"GATES PASSED {self.num_of_gates}")
                    if self.num_of_gates == 4: # if all gates are passed, take a small turn to the right to see the stop sign better
                        msg = String()
                        msg.data = "rightbig"
                        self.publication.publish(msg)
                        time.sleep(1.5) # stabilization time
                # Otherwise take only a small step
                else:
                    msg.data = "forward"
                    self.publication.publish(msg)
                    time.sleep(1.5) # stabilization time
        else:
            #### UPDATE THE ROTATION DIRECTION BASED ON THE RACING DAY GATE ARRANGEMENT!!!
            msg = String()
            msg.data = "rightsmall"
            self.publication.publish(msg)
            time.sleep(1.5) # stabilization time

        return True



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
