---
title: Lab 7
---

# Transforms

Today's scenario is about transforms. We will be calculating different poses in different coordinate systems and also work with roll pitch yaw representation. This page has 2 problems: Racing car and Dart-throwing, they are completely independent, so you can solve them in any order.

# Racing car problem

In this task we will use a racing car. But we won't drive it. The car joint's behaviour 
is already set (and always the same), and we know the simulation takes 5000 steps, so the car will move in exactly the same manner every time.
Our goal is to find such starting pose (position + orientation) of the car that it ends in a target gate.

The car is teleported to a given pose by the `build_world_with_car` function.

```python
#!/usr/bin/env python3

import pybullet as p
import pybullet_data
import math
import random

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

def make_boxes():
  pos_1 = [random.random(), random.random(), 0]
  angle = 3.1415 * 2 * random.random()
  pos_2 = [pos_1[0] + math.sin(angle) * 0.6, pos_1[1] - math.cos(angle) * 0.6, 0]
  return pos_1, pos_2

def build_world_with_car(pos):
  p.resetSimulation()
  p.setGravity(0, 0, -10)
  p.loadURDF("plane.urdf")
  car = p.loadURDF("racecar/racecar.urdf")
  p.resetBasePositionAndOrientation(car, pos[0], pos[1])
  return car

def simulate_car(car):
  inactive_wheels = [3, 5, 7]
  wheels = [2]
  for wheel in inactive_wheels:
    p.setJointMotorControl2(car, wheel, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
  steering = [4, 6]
  maxForce = 10
  targetVelocity = -2
  steeringAngle = 0.174
  steps = 5000
  for wheel in wheels:
    p.setJointMotorControl2(car,
                            wheel,
                            p.VELOCITY_CONTROL,
                            targetVelocity=targetVelocity,
                            force=maxForce)

  for steer in steering:
    p.setJointMotorControl2(car, steer, p.POSITION_CONTROL, targetPosition=steeringAngle)
  for i in range(steps):
     p.stepSimulation()
  return p.getBasePositionAndOrientation(car)

start_pose = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
car = build_world_with_car(start_pose)
end_pose = simulate_car(car)
pos_1, pos_2 = make_boxes()

# TODO: calculate calculated_pose
calculated_pose = start_pose

# The car should end its route with the front wheels between two boxes as in the example

car = build_world_with_car(calculated_pose)
p.loadURDF("cube.urdf", pos_1, globalScaling = 0.1)
p.loadURDF("cube.urdf", pos_2, globalScaling = 0.1)
simulate_car(car)
```

![Demo](/imgs/car_moving.gif)

Your goal is to calculate such `calculated_pos` value, that car route ends its route with the front wheels between two boxes. Your equations should be based on the following variables:

- start_pose - example starting pose
- end_pose - example finishing pose
- pos_1 and pos_2 - random positions of the boxes as generated by `make_boxes` function


---

# Dart-throwing problem

Now lets add some 3D rotations. You will need a 3D model [from here](https://free3d.com/3d-model/throwing-dart-v1--563436.html).
Also, please save following URDF as `dart.urdf` and copy the python code. Notice `--- EDIT ONLY BELOW THIS LINE ---` comment. You will find some instructions below that comment. Lets throw some darts!

```xml
<?xml version="1.0" ?>
<robot name="dart">
	<link name="baseLink">
		<inertial>
			<origin rpy="0 0 0" xyz="0 0 0"/>
			<mass value="1.0"/>
			<inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
		</inertial>
		<visual>
			<origin rpy="0 -1.5707963267948966 0" xyz="0 0 0"/>
			<geometry>
				<mesh filename="dart.obj" scale="0.1 0.1 0.1"/>
			</geometry>
			<material name="yellow">
				<color rgba="1 0.9 0 1"/>
			</material>
		</visual>
		<collision>
			<origin rpy="0 -1.5707963267948966 0" xyz="-0.5 0 0"/>
			<geometry>
				<box size="0.1 0.1 1"/>
			</geometry>
		</collision>
	</link>
</robot>
```

```python
#!/usr/bin/env python3

import numpy as np
import math
import pybullet as p
import pybullet_data
import random
import time

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

def make_target(pos):
	ret = p.loadURDF("sphere_small.urdf", pos, globalScaling = 2)
	p.changeVisualShape(ret, -1, rgbaColor = [1, 0.5, 0.7, 1])
	return ret

def rpy_to_dir(rpy):
	_, pitch, yaw = rpy
	x = math.cos(yaw) * math.cos(pitch)
	y = math.sin(yaw) * math.cos(pitch)
	z = -math.sin(pitch)

	v = np.array([x, y, z])
	norm = np.linalg.norm(v)
	return v / norm

def check_task_impl(dart_pose, target_pos, steps = None):
	if steps is None:
		steps = 500

	p.resetSimulation()
	p.loadURDF("plane.urdf")
	dart = p.loadURDF("dart.urdf")
	cube = make_target(target_pos)

	p.resetBasePositionAndOrientation(dart, dart_pose[0], dart_pose[1])
	p.resetBaseVelocity(dart, 1 * rpy_to_dir(p.getEulerFromQuaternion(dart_pose[1])))

	p.stepSimulation()
	if steps > 0:
		for i in range(steps):
			p.stepSimulation()
			time.sleep(0.01)
	else:
		while True:
			continue

def check_task0(yaw, steps = None):
	dart_pose = ([0, 0, 1], p.getQuaternionFromEuler([0, 0, yaw]))
	target_pos = [0, 1, 1]
	check_task_impl(dart_pose, target_pos, steps)

def check_task1(yaw, steps = None):
	dart_pose = ([0, 0, 1], p.getQuaternionFromEuler([0, 0, yaw]))
	target_pos = [1, 1, 1]
	check_task_impl(dart_pose, target_pos, steps)

def check_task2(pitch, steps = None):
	dart_pose = ([0, 0, 1], p.getQuaternionFromEuler([0, pitch, 0]))
	target_pos = [1, 0, 2]
	check_task_impl(dart_pose, target_pos, steps)

def check_task3(rpy, steps = None):
	dart_pose = ([0, 0, 1], p.getQuaternionFromEuler(rpy))
	target_pos = [1, -0.5, 2]
	check_task_impl(dart_pose, target_pos, steps)

def check_task4(rpy, steps = None):
	dart_pose = ([0, 0, 1], p.getQuaternionFromEuler(rpy))
	target_pos = [-1, -1, 0.2]
	check_task_impl(dart_pose, target_pos, steps)

def check_task5(rpy, steps = None):
	dart_pose = ([1, 0, 1], p.getQuaternionFromEuler(rpy))
	target_pos = [0, 0, 1]
	check_task_impl(dart_pose, target_pos, steps)

def check_task6(rpy, steps = None):
	dart_pose = ([-1, 1, 1], p.getQuaternionFromEuler(rpy))
	target_pos = [0, 0, 2]
	check_task_impl(dart_pose, target_pos, steps)

def check_task7(pitch, x, steps = None):
	dart_pose = ([x, 1, 1], p.getQuaternionFromEuler([0, pitch, math.radians(45)]))
	target_pos = [0, 1.5, 1.5]
	check_task_impl(dart_pose, target_pos, steps)

def check_task8(yaw, z, steps = None):
	dart_pose = ([-1, 1, z], p.getQuaternionFromEuler([0, math.radians(15), yaw]))
	target_pos = [0, -1, 1.1]
	check_task_impl(dart_pose, target_pos, steps if steps is not None else 1000)

def check_task9(pitch, steps = None):
	dart_pose = ([0, 0, 1], p.getQuaternionFromEuler([0, pitch, 0]))
	target_pos = [1.5, 0, 1.5]
	check_task_impl(dart_pose, target_pos, steps)

def check_task10(yaw, steps = None):
	dart_pose = ([0, 0, 1], p.getQuaternionFromEuler([0, 0, yaw]))
	target_pos = [-0.3, -1.3, 1]
	check_task_impl(dart_pose, target_pos, steps)

def check_task11(rpy, steps = None):
	dart_pose = ([0, 0, 1], p.getQuaternionFromEuler(rpy))
	target_pos = [1.2, 1.6, 1.7]
	check_task_impl(dart_pose, target_pos, steps if steps is not None else 1000)

def check_task12(yaw, z, steps = None):
	dart_pose = ([0.3, 0.8, z], p.getQuaternionFromEuler([0, math.radians(33), yaw]))
	target_pos = [0.5, -2, 1.8]
	check_task_impl(dart_pose, target_pos, steps if steps is not None else 1000)

def check_task13(x, z, steps = None):
	dart_pose = ([x, -1, z], p.getQuaternionFromEuler([0, math.radians(-15), math.radians(125)]))
	target_pos = [-0.4, 0.5, 1.2]
	check_task_impl(dart_pose, target_pos, steps)

# --- EDIT ONLY BELOW THIS LINE ---

# Demo, yaw 90 degrees is the solution, steps argument is optional
check_task0(math.radians(90), steps = 500)

# First batch of tasks, try to solve them by hand

#check_task1(yaw)
#check_task2(pitch)

#check_task3(rpy)
#check_task4(rpy)

#check_task5(rpy)
#check_task6(rpy)

#check_task7(pitch, x)
#check_task8(yaw, z)


# Second batch of tasks, do not hardcode anything, calculate required answers here, in python.

# Task9 is given as:
dart_pos = np.array([0, 0, 1])
target_pos = np.array([1.5, 0, 1.5])
# Calculate needed pitch.
#check_task9(pitch)

# Task10 is given as:
dart_pos = np.array([0, 0, 1])
target_pos = np.array([-0.3, -1.3, 1])
# Calculate needed yaw.
#check_task10(yaw)

# Task11 is given as:
dart_pos = np.array([0, 0, 1])
target_pos = np.array([1.2, 1.6, 1.7])
# Calculate needed rpy.
#check_task11(rpy)

# Task12 is given as:
dart_pos = np.array([0.3, 0.8, None])
target_pos = np.array([0.5, -2, 1.8])
pitch = math.radians(33)
# Calculate needed yaw and dart_pos[2].
#check_task12(yaw, z)

# Task13 is given as:
dart_pos = np.array([None, -1, None])
target_pos = np.array([-0.4, 0.5, 1.2])
rpy = np.array([0, math.radians(-15), math.radians(125)])
# Calculate needed dart_pos[0] and dart_pos[2].
#check_task13(x, z)
```


![Demo](/imgs/dart_throwing.gif)