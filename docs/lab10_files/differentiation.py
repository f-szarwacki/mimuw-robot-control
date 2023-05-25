from sympy import *
g, omega, theta, m_p, M, F, l, mu_p = symbols('g omega theta m_p M F l mu_p')
init_printing(use_unicode=True)

theta_dotdot = (g * sin(theta) + cos(theta) * ((-F - m_p * l * omega ** 2 * sin(theta)) / M) - (mu_p * omega) / (m_p * l)) / (l * (4/3 - (m_p * cos(theta) ** 2) / M))

x_dotdot = (F + m_p * l * (omega ** 2 * sin(theta) - theta_dotdot * cos(theta))) / M

theta_dotdot

x_dotdot


theta_dotdot.diff(theta).subs([(omega, 0), (theta, 0), (F, 0)])
theta_dotdot.diff(omega).subs([(omega, 0), (theta, 0), (F, 0)])
theta_dotdot.diff(F).subs([(omega, 0), (theta, 0), (F, 0)])

x_dotdot.diff(theta).subs([(omega, 0), (theta, 0), (F, 0)])
x_dotdot.diff(omega).subs([(omega, 0), (theta, 0), (F, 0)])
x_dotdot.diff(F).subs([(omega, 0), (theta, 0), (F, 0)])