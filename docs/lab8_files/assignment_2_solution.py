from assignment_2_lib import take_a_photo, drive

import numpy as np
import cv2


def get_ball_width_and_center(image):
    image = image[0:400, :, :]
    
    # conversion to HSV colorspace
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # positive red hue margin
    lower1 = np.array([0, 100, 50])
    upper1 = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv_image, lower1, upper1)

    # negative red hue margin
    lower2 = np.array([160,100,50])
    upper2 = np.array([189,255,255])
    mask2 = cv2.inRange(hsv_image, lower2, upper2)

    mask = mask1 + mask2

    width = np.max(np.sum(mask, axis=0))
    center_x = np.argmax(np.sum(mask, axis=0))

    return width, center_x

def get_poles_positions(image):
    # The poles are high and straight, we can use only the top of the picture.
    image = image[0:10, :, :] 

    # conversion to HSV colorspace
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # blue hue margin
    lower_blue = np.array([110, 100, 50])
    upper_blue = np.array([130, 255, 255])
    mask_blue = cv2.inRange(hsv_image, lower_blue, upper_blue)

    # A heuristic sufficient to guide the car - we check whether there is a pole
    # on the right of the photo and on the left, while there is no pole in the middle.
    middle_size = 60
    side_size = (640 - middle_size) // 2
    threshold = 0.1

    left_blue = np.mean(mask_blue[:, :side_size]) > threshold
    middle_blue = np.mean(mask_blue[:, side_size:side_size+middle_size]) > threshold
    right_blue = np.mean(mask_blue[:, side_size+middle_size:]) > threshold

    # This is approximately the middle of the goal, at least when the car is 
    # not far to the side.
    non_zero_indices = np.nonzero(np.sum(mask_blue, axis=0))[0]
    if non_zero_indices.size == 0:
        # Array is empty - poles are not visible.
        mean_of_indices_where_blue = None
    else:
        mean_of_indices_where_blue = int(np.nan_to_num(np.rint(np.mean(non_zero_indices))))

    return (left_blue, middle_blue, right_blue), mean_of_indices_where_blue


def ball_width_to_steps_distance(width, distance_to_ball_wanted=0.5):
    # Those constants have been found by experiments.
    # See the comment at the bottom of the file.
    distance_width_ratio = 31848.97959183673
    steps_distance_ratio = 2357.5263937301515

    distance = (1. / width) * distance_width_ratio
    steps = int(max(0, distance - distance_to_ball_wanted) * steps_distance_ratio)

    return steps, distance


def forward_distance(photo):
    photo = cv2.cvtColor(photo, cv2.COLOR_RGBA2BGR)
    
    width, _ = get_ball_width_and_center(photo)

    return ball_width_to_steps_distance(width)[0]


def find_a_ball(
    car, /, 
    save_starting_horizontal_offset=False, 
    direction_to_look_if_ball_not_visible=1
    ):
    
    how_many_drives_when_straight = 1
    how_many_steps_single_drive = 250
    straight_offset_tolerance = 25
    stop_distance = 0.89
    starting_horizontal_offset = 0
    saved = False

    while True:
        photo = take_a_photo(car)
        photo = cv2.cvtColor(photo, cv2.COLOR_RGBA2BGR)
        
        width, center_x = get_ball_width_and_center(photo)
        horizontal_offset = center_x - (photo.shape[1] // 2)

        # Check whether ball is visible. If not, set horizontal_offset
        # so that the car is turning into direction_to_look_if_ball_not_visible.
        if center_x == 0:
            horizontal_offset = -2 * direction_to_look_if_ball_not_visible * straight_offset_tolerance

        if not saved and save_starting_horizontal_offset:
            starting_horizontal_offset = horizontal_offset
            saved = True

        if abs(horizontal_offset) < straight_offset_tolerance:
            steps_needed, distance_left = ball_width_to_steps_distance(width, distance_to_ball_wanted=stop_distance)

            if distance_left < stop_distance:
                if save_starting_horizontal_offset:
                    return starting_horizontal_offset
                else:
                    return
            for _ in range(max(how_many_drives_when_straight, steps_needed // how_many_steps_single_drive)):
                drive(car, True, 0)
        else:
            drive(car, False, 0)
            turn = -1 if horizontal_offset > 0 else 1
            drive(car, True, turn)
     


def move_a_ball(car):
    # Look at the scene and check where the goal is - to the left 
    # or to the right. This should be the direction to look for the ball if
    # it is not visible.
    photo = take_a_photo(car)
    photo = cv2.cvtColor(photo, cv2.COLOR_RGBA2BGR)
    _, goal_center = get_poles_positions(photo)

    assert goal_center is not None, "Goal is not visible from starting position."

    goal_center_offset = goal_center - (photo.shape[1] // 2)

    turn_direction_after_ball_found = np.sign(find_a_ball(
        car, 
        save_starting_horizontal_offset=True,
        direction_to_look_if_ball_not_visible=-np.sign(goal_center_offset)
    ))

    # We look for the goal.
    photo = take_a_photo(car)
    photo = cv2.cvtColor(photo, cv2.COLOR_RGBA2BGR)

    _, goal_center = get_poles_positions(photo)

    # If goal is visible after we get to the ball, we can set the direction
    # based on that.
    if goal_center is not None:
        goal_center_offset = goal_center - (photo.shape[1] // 2)
        turn_direction_after_ball_found = -np.sign(goal_center_offset)

    # Turn until goal is straight.
    while True:
        photo = take_a_photo(car)
        photo = cv2.cvtColor(photo, cv2.COLOR_RGBA2BGR)

        blue_poles_position, _ = get_poles_positions(photo)

        left_blue, middle_blue, right_blue = blue_poles_position

        if not (left_blue and not middle_blue and right_blue):
            drive(car, True, 0) # This is needed to keep the ball in control.
            drive(car, True, turn_direction_after_ball_found)
        else:
            break

    # Go straight until blue goal is no longer seen. 
    drive_calls_per_poles_position_check = 5 
    while True:
        photo = take_a_photo(car)
        photo = cv2.cvtColor(photo, cv2.COLOR_RGBA2BGR)

        blue_poles_position, _ = get_poles_positions(photo)
        left_blue, middle_blue, right_blue = blue_poles_position

        if sum(blue_poles_position) > 0:
            for _ in range(drive_calls_per_poles_position_check):
                drive(car, True, 0)
        else:
            break
    
    # Finish to be sure to pass blue goal.
    finish_drives = 5
    for _ in range(finish_drives):
        drive(car, True, 0)




"""
Code used to determine constants used in width to steps/distance calculation.

from assignment_2_lib import *
from assignment_2_solution import *
from scipy.spatial import distance
import numpy as np
import matplotlib.pyplot as plt

NUMBER_OF_TESTS = 100

def calculate_distance_width_ratio():
    widths = []
    xs = np.linspace(1, 5, 50)
    for x in xs:
        car = build_world_with_car()
        pos_1 = [x, 0, 1]
        ball = p.loadURDF("sphere2red.urdf", pos_1, globalScaling=0.3)
        for _ in range(200):
            p.stepSimulation()
        photo = take_a_photo(car)
        _, width = forward_distance(photo)
        widths.append(width)
     
    return xs, widths, np.median(xs * widths)

def calculate_steps_distance_ratio():
    dsts = []
    dsts2 = []
    steps = np.linspace(2000, 15000, 10)
    for s in steps:
        car = build_world_with_car()
        base_care_position = p.getBasePositionAndOrientation(car)[0]
        simulate_car(car, 0, 2, int(s))
        dst = distance.euclidean(
            p.getBasePositionAndOrientation(car)[0],
            base_care_position,
        )
        dsts.append(dst)
        dst = distance.euclidean(
            p.getBasePositionAndOrientation(car)[0],
            (5, 0, 0),
        )
        dsts2.append(dst)
        
    return steps, dsts, dsts2, np.median(steps / dsts)


if __name__ == "__main__":
    
    xs, widths, distance_width_ratio = calculate_distance_width_ratio()
    
    plt.plot(widths, xs, label="xs = f(widths)")
    plt.plot(widths, xs * widths, label="xs * widths")
    
    print(f"{distance_width_ratio=}")
    
    steps, distances, distances2, steps_distance_ratio = calculate_steps_distance_ratio()
    print(f"{steps_distance_ratio=}")
    
    plt.plot(distances, steps, label='distances by steps')
    plt.plot(distances, steps / distances, label='steps/distances by distances')
    
    plt.legend()
    plt.show()

    cv2.waitKey()

"""