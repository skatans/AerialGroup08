# AerialGroup08

This uses https://github.com/clydemcqueen/tello_ros to communicate with the drone

#### Window 1 (tello drivers)
In tello_ros, after appropriate sourcing of the package:

```
ros2 launch tello_driver teleop_launch.py
```

#### Window 2 (command subscriber)
In the opencv_package, source the package and run:

```
ros2 run opencv_package cmd_sub
```

#### Window 3 (image subscription)
In the opencv_package, source the package and run:

```
ros2 run opencv_package img_sub
```

#### Window 4 (gate detection)
In the opencv_package, source the package and run:

```
ros2 run opencv_package gate_det
```
