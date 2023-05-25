# Tests are provided 'as is' and are not guaranteed to be correct.
# Only for private use

import importlib
import time

import solution
import unittest
import numpy as np

cartpole = importlib.import_module("cartpole-sim")


def my_test_solution(self, target=0, solution_name='Solution1'):
    # Create an instance of Solution
    if solution_name == 'Solution1':
        sol = solution.Solution1(self.get_state(), target)
    elif solution_name == 'Solution2':
        sol = solution.Solution2(self.get_state(), target)
    else:
        raise ValueError(f"Unknown solution name: {solution_name}")

    steps = 0
    total_effort = 0
    total_effort_sqr = 0
    while steps < cartpole.TEST_TIMEOUT_STEPS:
        # Call and apply the policy
        command = sol.update(self.get_state())
        cartpole.p.applyExternalForce(
            self.cartpole,
            cartpole.LINK_CART,
            forceObj=[0, command, 0],
            posObj=[0, 0, 0],
            flags=cartpole.p.WORLD_FRAME,
        )

        # Step pybullet
        self.apply_friction()
        cartpole.p.stepSimulation()
        time.sleep(cartpole.TIME_STEP_S)

        # Benchmarking
        total_effort += abs(command)
        total_effort_sqr += command ** 2
        steps += 1

        # Check for acceptance
        if self.stable_pos(target) and self.stable_angle(0): # Changed as stated on Slack.
            print(f"Accepting {self.get_state()}")
            return True, steps, total_effort, total_effort_sqr

    # Timeout occurred
    return False, None, None, None


def my_test_case(self, start_pos, start_angle, target, silent=False, solution_name='Solution1'):
    self.msg("SETUP")
    self.setup_state(start_pos, start_angle)

    self.msg("TESTING")
    ret = self.my_test_solution(target, solution_name)

    if not silent:
        if ret[0]:
            print(f"\n--- Success! ---\nSteps: {ret[1]}\nEffort: {ret[2]:.2f}\n")
            self.msg("SUCCESS", [0, 0.7, 0])
        else:
            print(f"\n--- Timeout! ---\n")
            self.msg("TIMEOUT", [0.7, 0, 0])
            raise Exception(
                f"Timeout! Failed test with parameters: start_pos={start_pos}, start_angle={start_angle}, target={target}")
        time.sleep(0.5)

    if not ret[0]:
        raise Exception(
            f"Timeout! Failed test with parameters: start_pos={start_pos}, start_angle={start_angle}, target={target}")

    return ret


def create_my_simulation():
    sim = cartpole.Simulation()
    sim.my_test_solution = my_test_solution.__get__(sim)
    sim.my_test_case = my_test_case.__get__(sim)
    return sim


def execute_and_increment(sim, counter, start_pos, start_angle, target, silent=False, solution_name='Solution1'):
    print(f'Testing {counter=} with parameters {start_pos=}, {start_angle=}, {target=}')
    try:
        sim.my_test_case(start_pos, start_angle, target, silent, solution_name)
        counter += 1
    except Exception as e:
        print(f"Error: {e}")
        print(f'Failed test with parameters: start_pos={start_pos}, start_angle={start_angle}, target={target}')
    return counter


class TestSolution(unittest.TestCase):
    # np.set_printoptions(formatter={'float': lambda x: "{0:0.18f}".format(x)})

    solution_name = 'Solution2'  # TODO change this to your own tests
    sim = create_my_simulation()

    def test_linearize_a(self):
        params = {
            'gravity': 9.8,
            'mass_cart': 1,
            'mass_pole': 0.1,
            'length_pole': 0.5,
            'mu_pole': 0.001,
        }
        A, B = solution.linearize(**params)
        target_A = np.array([[0.000000000000000000, 1.000000000000000000, 0.000000000000000000,
                                                     0.000000000000000000],
                                                    [0.000000000000000000, 0.000000000000000000, -0.717073170731707399,
                                                     0.002926829268292683],
                                                    [0.000000000000000000, 0.000000000000000000, 0.000000000000000000,
                                                     1.000000000000000000],
                                                    [0.000000000000000000, 0.000000000000000000, 31.551219512195125105,
                                                     -0.128780487804878052]])
        print(A, target_A, A - target_A, sep='\n')
        
        np.testing.assert_almost_equal(A, target_A, decimal=16)




    def test_linearize_b(self):
        params = {
            'gravity': 9.8,
            'mass_cart': 1,
            'mass_pole': 0.1,
            'length_pole': 0.5,
            'mu_pole': 0.001,
        }
        A, B = solution.linearize(**params)
        target_B = np.array([[0.000000000000000000],
                                                    [0.975609756097560954],
                                                    [0.000000000000000000],
                                                    [-2.926829268292682862]])

        print(B, target_B, B - target_B, sep='\n')

        np.testing.assert_almost_equal(B, target_B, decimal=16)

    '''def test_official_sim(self):
        counter = 0

        counter = execute_and_increment(self.sim, counter, 0.1, 0.03, 0, False, self.solution_name)
        counter = execute_and_increment(self.sim, counter, 1, 0, 0, False, self.solution_name)
        counter = execute_and_increment(self.sim, counter, -1.5, -0.4, 0, False, self.solution_name)
        counter = execute_and_increment(self.sim, counter, 1.5, -1, 0, False, self.solution_name)

        self.assertEqual(counter, 4)

    # Test might not work as implementing custom target was optional.
    def test_with_different_target(self):
        counter = 0

        counter = execute_and_increment(self.sim, counter, 0.1, 0.03, 1, False, self.solution_name)
        counter = execute_and_increment(self.sim, counter, 1, 0, -1, False, self.solution_name)

        self.assertEqual(counter, 2)

    # This test is debatable as probably the solutions won't be tested on such extreme values.
    # However, it is possible to pass this test.
    def test_out_of_boundaries(self):
        counter = 0

        counter = execute_and_increment(self.sim, counter, -1.5, -np.pi / 4, 0, False, self.solution_name)
        counter = execute_and_increment(self.sim, counter, -1.5, -np.pi / 2 + 0.2, 0, False, self.solution_name)

        counter = execute_and_increment(self.sim, counter, 1.5, -np.pi / 2 + 0.2, 0, False, self.solution_name)
        counter = execute_and_increment(self.sim, counter, 1.5, -np.pi / 2 + 0.2, 0, False, self.solution_name)

        self.assertEqual(counter, 4)

    def test_small_values(self):
        counter = 0
        np.random.seed(42)

        for i in range(5):
            start_pos = np.random.uniform(-0.5, 0.5)
            start_angle = np.random.uniform(-np.pi / 5, np.pi / 5)
            counter = execute_and_increment(self.sim, counter, start_pos, start_angle, 0, False, self.solution_name)

        self.assertEqual(counter, 5)

    def test_medium_values(self):
        counter = 0
        np.random.seed(42)

        for i in range(5):
            start_pos = np.random.uniform(-1, -0.5)
            start_pos *= np.random.choice([-1, 1])
            start_angle = np.random.uniform(-np.pi / 4, np.pi / 4)
            counter = execute_and_increment(self.sim, counter, start_pos, start_angle, 0, False, self.solution_name)

        self.assertEqual(counter, 5)

    def test_large_values(self):
        counter = 0
        np.random.seed(42)

        for i in range(5):
            start_pos = np.random.uniform(-2.0, -1.5)
            start_pos *= np.random.choice([-1, 1])
            start_angle = np.random.uniform(np.pi / 3, np.pi / 2)
            start_angle *= -np.sign(start_pos)
            counter = execute_and_increment(self.sim, counter, start_pos, start_angle, 0, False, self.solution_name)

        self.assertEqual(counter, 5)
    '''

if __name__ == '__main__':
    # BY DEFAULT RUNS WITH SOLUTION1. YOU CAN CHANGE IT TO SOLUTION2 IN TESTCASE ABOVE - solution_name = 'Solution1' or
    # solution_name = 'Solution2'
    print(f'Starting testing with solution {TestSolution.solution_name}')
    unittest.main()