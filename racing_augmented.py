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

flying = False

plywood_light = [(10, 30), (0,40), (220,240)]
plywood_mid = [(10, 30), (10,100), (200,240)]
plywood_dark= [(5, 30), (20,100), (130,240)]

red_area = 0
red_x1 = 0
red_x2 = 0
red_y1 = 0
red_y2 = 0

num_of_gates = 0
take_off = False

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
        print("model inited...")

    def service_response_callback(self, future):
        try:
            response = future.result()
            print(f"Service call succeeded: {response}")
        except Exception as e:
            print(f"Service call failed: {e}")

    def listener_callback(self, data):
        global num_of_gates
        global take_off
        
        self.get_logger().info('Receiving video frame')
        if self.processing_image:
            return
        self.processing_image = True

        self.get_logger().info('Receiving video frame')
        frame = self.br.imgmsg_to_cv2(data, 'bgr8')
        
        if take_off is False:
            while not self.client.wait_for_service(timeout_sec=1.0):
                print('Service not available, waiting...')
            request = TelloAction.Request()
            self.get_logger().info('Takeoff command')
            request.cmd = 'takeoff'
            future = self.client.call_async(request)
            future.add_done_callback(self.service_response_callback)
            take_off = True
        elif self.detect_gate(frame) is False:
            area = self.detect_stop(frame)
            if area > 3500:                                                ### THRESHOLD FOR A BIG RED AREA
                while not self.client.wait_for_service(timeout_sec=1.0):
                    print('Service not available, waiting...')
                request = TelloAction.Request()
                self.get_logger().info('Land command')
                request.cmd = 'land'
                future = self.client.call_async(request)
                future.add_done_callback(self.service_response_callback)

        self.processing_image = False
    
    def detect_gate(self, image):
        global num_of_gates
        if num_of_gates >= 4:
            print(f"########All gate passed {num_of_gates}")
            return False
        
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
            print(f">>>>>>>>>>aligned {aligned}")
            # Move forward if alignment is ok
            if aligned == True:
                msg = String()
                # If the radius of the gate is large enough, move a lot forward
                print(f">>>>>{x2-x1}    {0.9*image_height}")
                if (x2-x1) > 0.9*image_height:
                    msg.data = "forwardlong"
                    self.publication.publish(msg)
                    time.sleep(2.5) # stabilization time
                    num_of_gates = num_of_gates + 1
                    print(f"GATES PASSED {num_of_gates}")
                    if num_of_gates == 4: # if all gates are passed, take a small turn to the right to see the stop sign better
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


    def detect_stop(self, image):
        global red_x1
        global red_x2
        global red_y1
        global red_y2
        global red_area

        frame = image
        image_height = frame.shape[0]
        image_width = frame.shape[1]
    
        blur = cv2.GaussianBlur(frame,(15,15),0)
        #cv2.waitKey(1)

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
        print("first")
        if white_ratio > 0.75:
            return 1
    
        # red mask
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 180])
        lower_red2 = np.array([170, 70, 50])
        upper_red2 = np.array([180, 255, 180])

        # Convert red image and mask back to BGR for OpenCV display
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask_red1, mask_red2)

        # Apply the red mask to the image
        red_result = cv2.bitwise_and(frame, frame, mask=red_mask)
        cv2.imwrite("re.jpg", red_result)

        red_edges = cv2.Canny(red_result, 50, 150)
        # Find contours in the edge-detected image
        contours, _ = cv2.findContours(red_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #red_contours, _ = cv2.findContours(red_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Check for a big red area in the contours
        print("SECOND")
        
        if contours:
    	    red_contours = max(contours, key=cv2.contourArea)
    	    moments = cv2.moments(red_mask)
        else:
    	    return 0
        print(f">>>>>>>???{cv2.boundingRect(red_contours)}")
    
        #for contour in red_contours:
        for i in range(1):
            print("THIRD")
            area = cv2.contourArea(red_contours)
            red_area = area
            red_x1, red_y1, w, h = cv2.boundingRect(red_contours)
            red_x2 = red_x1 + w
            red_y2 = red_y1 + h
            print(f">>>>>>>{cv2.boundingRect(red_contours)}")
            cv2.rectangle(frame, (red_x1, red_y1), (red_x2, red_y2), (0, 255, 0), 2)
            #cv2.imwrite("hei red.jpg", frame)
            #cv2.imshow("Red Mask Applied", frame)
            print(f">>>>>showed")
            #cv2.waitKey(1)
            #cv2.putText(red_result, f"Area: {area}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            print(f"area>> {area}")
            if area > 3500:  # Threshold for a "big" red area
                print(f"area>> {area}")
                x, y, w, h = cv2.boundingRect(red_contours)
                ####################test
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "Big Red Area", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                #cv2.imshow("Red Mask Applied", red_result)
                #cv2.waitKey(1)
                return area
        
        # ALIGNMENT START
##########
        x1, y1, x2, y2, area = red_x1, red_y1, red_x2, red_y2, red_area
        # Draw a red dot in the center of the largest gate
        center_x_gate = (x1 + x2) // 2
        center_y_gate = (y1 + y2) // 2
        cv2.circle(frame, (center_x_gate, center_y_gate), 5, (0, 0, 255), -1)  #
        # Draw larggest gate bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        self.publisher.publish(self.br.cv2_to_imgmsg(frame))
        
        # Align the gate
        image_center_height = image_height / 3
        image_center_width = image_width / 2

        error_height = center_y_gate - image_center_height
        error_width = center_x_gate - image_center_width

        height_threshold = image_height / 6
        width_threshold = image_width / 6

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
            # If the radius of the gate is large enough, move forward
            msg.data = "forward"
            self.publication.publish(msg)
            time.sleep(1.5) # stabilization time
##########
        
        return 0


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
