# Graded Homework 3

This is a third graded homework. Deadline is 18th January 08:00.
At the end of this scenario you will find more details and the exact grading scheme.

# Updates (newest first)

## 10.01.2022

Unfortunately, real hardware cartpole rigs presented many problems and it was impossible to make them work in time.
Fortunately, points allocation was designed in a way that will not suffer due to this issue.
Apologies for the situation and if machines are ready during this semester we will try to organize a demo of the system.

## 09.01.2022

There was a mistake in check for acceptance of the solution in that it was not checking for correct angle being set.
It allowed to pass the tests regardless of the pole position, so in this aspect it is definitely fixed now.
In terms of being more restrictive to pass the tests this probably does not change anything,
because in all solutions we saw `stable_angle` comes earlier than `stable_pos`.
Regardless, here is the change and **it will be in force for automatic grading**:

![image](https://user-images.githubusercontent.com/7950377/212488835-24b0e008-4d3b-4b71-a7b2-e06c3fd25274.png)


## 28.12.2022

1. Slack channel [#rc-hw3](https://machinelearni-yqf6662.slack.com/archives/C04GKAQM4CS) contains some questions and answers.
Please join the channel so that you do not miss anything.

2. There was a discrepancy between the specification on this page and a print in the code.
The print is informational only and grading will be probably executed with `silent=True` anyway.
The specification on this page is therefore binding, which means `res[3]` does not matter for grading and `res[2]` does.
For clarity the code was updated to print `res[2]`:

    ![image](https://user-images.githubusercontent.com/7950377/209737064-61c73c13-bc38-4ab1-ae34-74121a88c4f9.png)


# General info

In this assignment, you will be working on a classic control problem known as the cart-pole system.
The goal of the assignment is to design and implement a controller for the cart-pole system using the provided simulation environment.
You will have the opportunity to apply the control theory concepts that you have learned in this course,
as well as gain experience in implementing and testing control algorithms in a simulated environment.

<s>
In January, there will be a bonus opportunity to further demonstrate your skills by implementing your controller on real hardware.
The first lab session of the month will be dedicated to working with these physical cart-pole machines.
Successfully completing this bonus portion of the assignment will allow you to earn more than 100% of the total points for the assignment.
</s>

# Cart-pole problem

The cart-pole system is a classic benchmark for control algorithms because it involves a balance between complexity and simplicity.
The system consists of a cart that can move horizontally along a track and a pole that is attached to the cart by a hinge.
The goal of the controller is to balance the pole vertically on the cart, which requires it to continuously adjust
the position and velocity of the cart to counter the movement of the pole.

However, this task is made more challenging by the nonlinear dynamics of the system. The motion of the pole is
influenced by factors such as its angle and angular velocity, as well as the position and velocity of the cart. As a result,
the controller must be able to adapt to these changing conditions in order to effectively balance the pole.

https://user-images.githubusercontent.com/7950377/209263837-02b1942a-028c-430d-9560-4fa362782ae5.mp4

# Task

You are provided with the 4-dimensional state of the cart-pole system, consisting of position, velocity, angle, and angular velocity,
as well as the formulas used in the openai gym code to define the transitions between states.

Your task is to design and implement a controller for the cart-pole system that is able to stabilize
the system around the fixed point (0,0,0,0). This will involve balancing the pole vertically and maintaining
the position, velocity, angle, and angular velocity of the system at or near zero. The controller
should be able to handle perturbations and maintain stability even when the system is close to departing from this fixed point.

https://user-images.githubusercontent.com/7950377/209263944-5e749511-2eff-4f76-8a34-65bee13c0a8a.mp4

Specifically, you will need to:

1. Linearize the non-linear model of the cart-pole system around the fixed point (0,0,0,0).
You are given the formulas for the dynamics of the system, which are available
[here](https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py#L139-L147).
You should implement a function `linearize` that produces the linearization using parameters such as
the masses of the objects or the coefficient of friction (described
[here](https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py#L140)).
This function will be checked automatically during the grading phase, so please stick to the signature provided
in the example `solution.py` file. Notice that it is not required to show pen&paper working out of the linearization and
therefore you should put extra effort to polish your `linearize` function so that it is clear how does it work.
2. Use the Python "control" library, available [here](https://python-control.readthedocs.io/) (`pip install control`),
as a black box to implement LQR. You should manipulate the parameters of the LQR cost function to optimize for:
    - Case 1: Minimum time to stabilization (measured in simulation steps)
    - Case 2: Minimum energy expenditure (measured as the sum of the absolute values of the control input, u)
while ensuring that the system is stabilized within a fixed number of timesteps (see the code for limits)
    
    Both cases will be graded.
    

As with the previous assignment, you can obtain all of the necessary code for this assignment by cloning this repository:

```bash
git clone https://github.com/mim-uw/rc-2022-23
cd rc-2022-23/lab10_files
```

All of the relevant files for this assignment can be found in the `lab10_files` directory.
All relevant numbers, limits and physical parameters should be found in these files,
meaning reading both python and URDF code is meant to be a part of the assignement.

# Submiting

Submiting will be done through the moodle page.
You should submit just 1 file: `solution.py`. Make sure you used the provided template for compatibility with automatic grading.
Deadline is 18th January 08:00.

# Scoring

Points for this task will be distributed as follows <s>(intentionally sums to 110%)</s>:

- 30% — Correctness of the `linearize` function, passing automated tests
- 30% — Time optimized controller (Case 1): correct LQR reasoning, passing automated tests
- 30% — Energy optimized controller (Case 2): correct LQR reasoning, passing automated tests
- 10% — An overall impression (selection of methods, solutions performance, source code quality, …)
- <s>10% — Successfully running your controller on the real hardware during lab10</s>
