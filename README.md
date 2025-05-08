# AerialGroup08

This uses https://github.com/clydemcqueen/tello_ros to communicate with the drone

#### Window 1 (tello drivers)
In tello_ros, after appropriate sourcing of the package:

```
ros2 launch tello_driver teleop_launch.py
```

#### Window 2 (command subscriber)
This node captures commands from tha main node and sends them to the drone

In the opencv_package, source the package and run:

```
ros2 run opencv_package cmd_sub
```

#### Window 3 (image subscription)
Receives images from gate detection and shows the bounding boxes and aiming dot in the image

In the opencv_package, source the package and run:

```
ros2 run opencv_package img_sub
```

#### Window 4 (gate detection)
The main gate detection node. Reads the image coming from the drone, sends commands to the command subscriber and publishes image.

In the opencv_package, source the package and run:

```
ros2 run opencv_package gate_det
```
