from djitellopy import Tello
import torch
import cv2
import time
import socket
import subprocess
import warnings

model = None
frame = None
tello = None

def start_play_video():
    # Define the ffmpeg command as a list of arguments
    ffmpeg_command = [
	    "ffmpeg",
	    "-f", "mjpeg",
	    "-i", "udp://127.0.0.1:5001",
	    "-pix_fmt", "yuv420p",
	    "-f", "sdl",
	    "Video Display"
    ]

    # Run the command in the background
    process = subprocess.Popen(ffmpeg_command)

def send_image_tcp(image, udp_ip="127.0.0.1", udp_port=5001):
    CHUNK_SIZE = 65000  # Reduce chunk size for smoother UDP transfer
    _, img_encoded = cv2.imencode('.jpg', image)
    img_bytes = img_encoded.tobytes()

    total_size = len(img_bytes)
    num_chunks = (total_size // CHUNK_SIZE) + (1 if total_size % CHUNK_SIZE != 0 else 0)

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        for i in range(num_chunks):
            start_idx = i * CHUNK_SIZE
            end_idx = min((i + 1) * CHUNK_SIZE, total_size)
            chunk = img_bytes[start_idx:end_idx]
            bytes_sent = sock.sendto(chunk, (udp_ip, udp_port))  # Send chunk
            if bytes_sent != len(chunk):
                print(f"Warning: Not all data sent! Sent {bytes_sent} out of {len(chunk)} bytes.")
            #print(f"Sent chunk {i + 1}/{num_chunks} of size {len(chunk)} bytes")
    
        #print(f"Sent total {num_chunks} chunks.")

def drone_forward(dist):
    print("forward")

def drone_init():
    # Initialize and connect to Tello
    global tello
    tello = Tello()
    tello.connect()
    print(f"#####Battery: {tello.get_battery()}%")
    
    # Start video stream
    tello.streamon()
    #######frame_read = tello.get_frame_read()
    # Takeoff
    tello.takeoff()
    time.sleep(4) 
    for i in range(5):
        test = tello.get_frame_read()
    print("#####drone inited")

def model_init():
    global model
    # Load custom YOLOv5 model
    model = torch.hub.load('./ultralytics', 'custom', source='local', path='./ultralytics/weights/bests640e75n1010.pt', force_reload=True)
    model.eval()
    print("#####model inited")

def frame_from_img(file):
    global frame
    frame = cv2.imread(file)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return img_rgb

def frame_from_drone():
    print("global frame from drone reading ...........................")
    global tello
    global frame
    frame_read = tello.get_frame_read()
    frame = frame_read.frame
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return img_rgb

def get_largest_gate(img_rgb):
    results = model(img_rgb)
    detections = results.xyxy[0]
    
    #draw result
    for *box, conf, cls in detections:
        x1, y1, x2, y2 = map(int, box)
        label = model.names[int(cls)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    
    largest_gate = None
    largest_area = 0
    # Loop through detections and find the largest gate
    for *box, conf, cls in detections:
        label = model.names[int(cls)]


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
        # Draw a green dot in the center of the image
        center_x_image = frame.shape[1] // 2
        center_y_image = frame.shape[0] // 2
        cv2.circle(frame, (center_x_image, center_y_image), 5, (0, 255, 0), -1)
        # Draw direction line
        cv2.arrowedLine(frame, (center_x_gate, center_y_gate), (center_x_image, center_y_image), (0, 0, 255), 3, tipLength=0.2)

        print(f"#####{largest_gate}, frame {frame.shape}, area ratio {area / (frame.shape[1] * frame.shape[0])}")
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3) 
        cv2.imwrite("out.jpg", frame)
        send_image_tcp(frame)
        #cv2.imshow("Frame with Gate", frame)
    return largest_gate

def move_closeto_gate(largest_gate):
    global tello
    x1, y1, x2, y2, area = largest_gate
    ratio = area / (frame.shape[1] * frame.shape[0])
    print("##move close to gate ratio {ratio}")
    if ratio < 0.08:
        tello.move_forward(50)
    elif ratio < 0.2:
        tello.move_forward(30)
    elif ratio < 0.3:
        tello.move_forward(25)
    elif ratio < 0.42:
        tello.move_forward(20)
    else:
        print("#moving through the gate")
        tello.move_forward(120)

def centerize_gate(largest_gate):
    global tello
    print("##centerize_gate")
    x1, y1, x2, y2, area = largest_gate
    center_x_frame = frame.shape[1] // 2
    center_y_frame = frame.shape[0] // 2
    center_x_gate = (x1 + x2) // 2
    center_y_gate = (y1 + y2) // 2
    ratio = area / (frame.shape[1] * frame.shape[0])
    if ratio > 0.38:
        return
    #horizontal
    if center_x_frame > center_x_gate: #move left
        if min(x1, x2) < 50:
            tello.move_left(20)
    elif center_x_frame < center_x_gate: #move right
        if max(x1, x2) > frame.shape[1] - 50:
            tello.move_right(20)
    #vertical
    if center_y_frame > center_y_gate: #move up
        if min(y1, y2) < 50:
            tello.move_up(20)
    elif center_y_frame < center_y_gate: #move down
        if max(y1, y2) > frame.shape[0] - 40:
            tello.move_down(20)
        
def possible_landing():
    #tello.land()
    return None

warnings.simplefilter("ignore", category=FutureWarning)
start_play_video()
drone_init()
model_init()

"""
for i in range(1, 101):
     file = file_name = f"validationImg/{i:08d}.jpg"
     img_rgb = frame_from_img(file)
     largest_gate = get_largest_gate(img_rgb)
     print(f"################{largest_gate}")
     time.sleep(0.1)
"""
for i in range(1, 10000):
    print(f"################")
    img_rgb = frame_from_drone()
    largest_gate = get_largest_gate(img_rgb)
    centerize_gate(largest_gate)
    move_closeto_gate(largest_gate)
    possible_landing()
    time.sleep(0.1)

# Clean up
tello.streamoff()

