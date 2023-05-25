import control
import numpy as np


# This is a stub of your solution
# Add your code in any organized way, but please keep the following signatures unchanged
# Solution1 should optimize for speed, Solution2 for effort. Refer to the assignement specification.


# Keep this signature unchanged for automated testing!
# Returns 2 numpy arrays - matrices A and B
def linearize(
    gravity: float,
    mass_cart: float,
    mass_pole: float,
    length_pole: float,
    mu_pole: float,
):

    A = np.zeros((4, 4))
    B = np.zeros((4, 1))

    """
Formulas for the theta_dotdot and x_dotdot:
Note: we assume mu_c to be 0.
https://coneural.org/florian/papers/05_cart_pole.pdf

Derivation using sympy:

>>> from sympy import *
>>> g, omega, theta, m_p, M, F, l, mu_p = symbols('g omega theta m_p M F l mu_p')
>>> init_printing(use_unicode=True)
>>> theta_dotdot = (g * sin(theta) + cos(theta) * ((-F - m_p * l * omega ** 2 * sin(theta)) / M) - (mu_p * omega) / (m_p * l)) / (l * (4/3 - (m_p * cos(theta) ** 2) / M))
>>> x_dotdot = (F + m_p * l * (omega ** 2 * sin(theta) - theta_dotdot * cos(theta))) / M
>>> theta_dotdot
                  ⎛           2       ⎞       
           μₚ⋅ω   ⎝-F - l⋅mₚ⋅ω ⋅sin(θ)⎠⋅cos(θ)
g⋅sin(θ) - ──── + ────────────────────────────
           l⋅mₚ                M              
──────────────────────────────────────────────
        ⎛                         2   ⎞       
        ⎜                   mₚ⋅cos (θ)⎟       
      l⋅⎜1.33333333333333 - ──────────⎟       
        ⎝                       M     ⎠       
>>> x_dotdot
         ⎛            ⎛                  ⎛           2       ⎞       ⎞       ⎞
         ⎜            ⎜           μₚ⋅ω   ⎝-F - l⋅mₚ⋅ω ⋅sin(θ)⎠⋅cos(θ)⎟       ⎟
         ⎜            ⎜g⋅sin(θ) - ──── + ────────────────────────────⎟⋅cos(θ)⎟
         ⎜ 2          ⎝           l⋅mₚ                M              ⎠       ⎟
F + l⋅mₚ⋅⎜ω ⋅sin(θ) - ───────────────────────────────────────────────────────⎟
         ⎜                         ⎛                         2   ⎞           ⎟
         ⎜                         ⎜                   mₚ⋅cos (θ)⎟           ⎟
         ⎜                       l⋅⎜1.33333333333333 - ──────────⎟           ⎟
         ⎝                         ⎝                       M     ⎠           ⎠
──────────────────────────────────────────────────────────────────────────────
                                      M                                       

>>> theta_dotdot.diff(theta).subs([(omega, 0), (theta, 0), (F, 0)])
            g            
─────────────────────────
  ⎛                   mₚ⎞
l⋅⎜1.33333333333333 - ──⎟
  ⎝                   M ⎠
>>> theta_dotdot.diff(omega).subs([(omega, 0), (theta, 0), (F, 0)])
             -μₚ             
─────────────────────────────
 2    ⎛                   mₚ⎞
l ⋅mₚ⋅⎜1.33333333333333 - ──⎟
      ⎝                   M ⎠
>>> theta_dotdot.diff(F).subs([(omega, 0), (theta, 0), (F, 0)])
            -1             
───────────────────────────
    ⎛                   mₚ⎞
M⋅l⋅⎜1.33333333333333 - ──⎟
    ⎝                   M ⎠
>>> x_dotdot.diff(theta).subs([(omega, 0), (theta, 0), (F, 0)])
          -g⋅mₚ          
─────────────────────────
  ⎛                   mₚ⎞
M⋅⎜1.33333333333333 - ──⎟
  ⎝                   M ⎠
>>> x_dotdot.diff(omega).subs([(omega, 0), (theta, 0), (F, 0)])
             μₚ            
───────────────────────────
    ⎛                   mₚ⎞
M⋅l⋅⎜1.33333333333333 - ──⎟
    ⎝                   M ⎠
>>> x_dotdot.diff(F).subs([(omega, 0), (theta, 0), (F, 0)])
                mₚ           
1 + ─────────────────────────
      ⎛                   mₚ⎞
    M⋅⎜1.33333333333333 - ──⎟
      ⎝                   M ⎠
─────────────────────────────
              M              

The remaining derivatives are trivially either 0 or 1.

    """

    overall_mass = mass_pole + mass_cart

    A[0, 1] = 1
    
    A[1, 2] = (-gravity * mass_pole) / (overall_mass * (4/3 - mass_pole / overall_mass))
    A[1, 3] = mu_pole / (length_pole * (overall_mass * 4/3 - mass_pole))

    A[2, 3] = 1
    
    A[3, 2] = gravity / (length_pole * (4/3 - mass_pole / overall_mass))
    A[3, 3] = -mu_pole / (mass_pole * length_pole ** 2 * (4/3 - mass_pole / overall_mass))

    B[1, 0] = (1 + mass_pole / (overall_mass * (4/3 - mass_pole / overall_mass))) / overall_mass

    B[3, 0] = -1 / (overall_mass * length_pole * (4/3 - mass_pole / overall_mass))

    return A, B


class Solution1:
    # Keep this signature unchanged for automated testing!
    # Reminder: implementing arbitrary target_pos is not required, but please try!
    def __init__(self, init_state, target_pos, Q=None, R=None):
        params = {
            'gravity': 9.8,
            'mass_cart': 1,
            'mass_pole': 0.1,
            'length_pole': 0.5,
            'mu_pole': 0.001,
        }
        self.A, self.B = linearize(**params)
        self.Q = 20 * np.eye(4) if Q is None else Q
        self.R = np.eye(1) if R is None else R

        self.K, _, _ = control.lqr(self.A, self.B, self.Q, self.R)
        

    # Keep this signature unchanged for automated testing!
    # Returns one float - a desired force (u)
    def update(self, state):
        #print(self.K, state)
        return float(-self.K @ state)


class Solution2:
    # Keep this signature unchanged for automated testing!
    # Reminder: implementing arbitrary target_pos is not required, but please try!
    def __init__(self, init_state, target_pos, Q=None, R=None):
        params = {
            'gravity': 9.8,
            'mass_cart': 1,
            'mass_pole': 0.1,
            'length_pole': 0.5,
            'mu_pole': 0.001,
        }
        self.A, self.B = linearize(**params)
        self.Q = np.eye(4) if Q is None else Q
        self.R = 10 * np.eye(1) if R is None else R

        self.K, _, _ = control.lqr(self.A, self.B, self.Q, self.R)

    # Keep this signature unchanged for automated testing!
    # Returns one float - a desired force (u)
    def update(self, state):
        return float(-self.K @ state)
