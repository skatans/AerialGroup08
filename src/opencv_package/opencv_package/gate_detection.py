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
from djitellopy import Tello # test

import time

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
        '''
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
        '''
        if self.num_of_gates == 4:
            if self.detect_stop_signal(frame):
                while not self.client.wait_for_service(timeout_sec=1.0):
                    print('Service not available, waiting...')

                request = TelloAction.Request()
                self.get_logger().info('Sending land command')
                request.cmd = 'land'
                #self.flying = False

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

        if yellow_ratio > 0.75:
            return 1
        return 0

   # Check for the stop sign, and navigate towards it
    def find_stop_signal(self, image):
        frame = image

        image_height = frame.shape[0]
        image_width = frame.shape[1]

        # Image centerpoint
        image_center_height = image_height // 2
        image_center_width = image_width // 2

        # How large area is considered to be midpoint good enough
        height_threshold = image_height / 5
        width_threshold = image_width / 5

        red_mask = self.create_mask(frame, 'red')

        # Calculate the mass center of the red area
        red_ratio = cv2.countNonZero(red_mask) / (red_mask.shape[0] * red_mask.shape[1])
        frame = cv2.bitwise_and(frame, frame, mask=red_mask)

        # Draw a rectangle with the center being the image center and the sides being error radius away
        top_left = (int(image_center_width - width_threshold), int(image_center_height - height_threshold))
        bottom_right = (int(image_center_width + width_threshold), int(image_center_height + height_threshold))
        cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 2)
        cv2.putText(frame, "Error Radius", (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Draw the center of the image on the frame
        cv2.circle(frame, ((image_center_width), (image_center_height)), 5, (0, 0, 255), -1)
        cv2.putText(frame, f"Image center: ({image_center_width}, {image_center_height})", ((image_center_width) + 10, (image_center_height) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 2)

        cv2.imshow("Center", frame)

        if red_ratio < 0.004:
            #self.get_logger().info('Red area too small, skipping center calculation')
            return

        else:
            #self.get_logger().info('Red area ok')

            moments = cv2.moments(red_mask)
            if moments["m00"] != 0:
                self.get_logger().info('Finding center')
                center_x = int(moments["m10"] / moments["m00"])
                center_y = int(moments["m01"] / moments["m00"])

                # Draw the center of the red area on the frame
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                cv2.putText(frame, f"Red center: ({center_x}, {center_y})", (center_x + 10, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 2)

                cv2.imshow("Center", frame)
            else:
                center_x, center_y = None, None
                return

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
                elif abs(error_width) > width_threshold:
                    aligned = False
                    self.check_right_left(error_width, width_threshold)

                # If the red area is aligned, move forward
                if aligned:
                    self.get_logger().info('Aligned with red area, send forward')
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
        return
        frame = image
        image_height = frame.shape[0]
        image_width = frame.shape[1]
        blur = cv2.GaussianBlur(frame,(15,15),0)
        cv2.waitKey(1)
        gate_detected = False

        image_center_height = image_height / 2
        image_center_width = image_width / 2

        height_threshold = image_height / 10
        width_threshold = image_width / 10

        '''FILTERING'''
        # green mask
        red_mask = self.create_mask(frame, 'red')
        green_mask = self.create_mask(frame, 'green')

        # Compare red and green areas
        self.detected_red_size = cv2.countNonZero(red_mask)
        self.detected_green_size = cv2.countNonZero(green_mask)

        # If red areas are larger than green areas, check for the stop sign
        if self.detected_red_size > self.detected_green_size:
            edges = cv2.Canny(red_mask, 50, 150)
            # Find contours in the edge-detected image
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Calculate the center of the red area
            moments = cv2.moments(red_mask)
            if moments["m00"] != 0:
                center_x = int(moments["m10"] / moments["m00"])
                center_y = int(moments["m01"] / moments["m00"])
                #cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                #cv2.putText(frame, f"Center: ({center_x}, {center_y})", (center_x + 10, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            else:
                center_x, center_y = None, None

            if center_x is not None and center_y is not None:


                error_height = center_x - image_center_height
                error_width = center_y - image_center_width
                aligned = True
                if abs(error_height) > height_threshold:
                    aligned = False
                    self.check_up_down(error_height, height_threshold)
                if abs(error_width) > width_threshold:
                    aligned = False
                    self.check_right_left(error_width, width_threshold)

                if aligned:
                    print("aligned with red area")
                    msg = String()
                    msg.data = "forward"
                    self.publication.publish(msg)
                    time.sleep(1.5) # stabilization time
        else:
            edges = cv2.Canny(green_mask, 50, 150)
            # Find contours in the edge-detected image
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
            # Find the largest circle and midpoints of all circles
            largest_circle = None
            largest_radius = 0
            circle_midpoints = []

            # Draw contours on the original frame
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 2000:
                    continue

                # Fit a minimum enclosing circle around the contour
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(radius)

                if radius < 40:
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
                    cv2.putText(green_mask, "danger", (x, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            
                # Draw the contour and label the shape
                cv2.drawContours(green_mask, [approx], -1, (0, 255, 0), 2)
                x, y, w, h = cv2.boundingRect(approx)
                cv2.putText(green_mask, shape, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                # Draw the bounding rectangle
                cv2.rectangle(green_mask, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Draw the largest circle
            if largest_circle:
                center, radius = largest_circle
                if radius > 200:
                    gate_detected = True
                    cv2.circle(green_mask, center, radius, (0, 255, 255), 2)
                    cv2.putText(green_mask, f"Radius: {radius}", (center[0] - 40, center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    cv2.imshow("Largest circle", res)
                    cv2.waitKey(1)

                    error_height = center[1] - image_center_height
                    error_width = center[0] - image_center_width

                    # Align the gate
                    aligned = True
                    if abs(error_height) > height_threshold:
                        aligned = False
                        self.check_up_down(error_height, height_threshold)
                    if abs(error_width) > width_threshold:
                        aligned = False
                        self.check_right_left(error_width, width_threshold)

                    if aligned:
                        print("aligned")
                        msg = String()
                        # If the radius of the gate is large enough, move a lot forward
                        if radius*2 > 0.9*image_height:
                            msg.data = "forwardlong"
                            self.publication.publish(msg)
                            time.sleep(2.5) # stabilization time
                        # Otherwise take only a small step
                        else:
                            msg.data = "forward"
                            self.publication.publish(msg)
                            time.sleep(1.5) # stabilization time

            if not gate_detected:
                print("searching")
                msg = String()
                aligned = False
                # Calculate the horizontal center of mass of the green mask
                moments = cv2.moments(green_mask)
                if moments["m00"] != 0:
                    center_x = int(moments["m10"] / moments["m00"])
                else:
                    center_x = 0  # Default to left if no green detected

                # Determine if most of the white is on the left or right
                if center_x < image_width // 2:
                    msg.data = "left"
                else:
                    msg.data = "right"
                self.publication.publish(msg)
                time.sleep(1) # stabilization time

            # Draw all circle midpoints
            for center, radius in circle_midpoints:
                cv2.circle(frame, center, 3, (255, 0, 0), -1)
                cv2.putText(frame, f"({center[0]}, {center[1]})", (center[0] + 5, center[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)


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