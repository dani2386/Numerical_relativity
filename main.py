from src.WaveEquationSolver import WaveEquationSolver
import numpy as np


n = 120
m = 100
delta_x = 1
delta_t = 0.5


def f(x):
    return np.exp(-pow(x - 20, 2)/(2 * pow(2 , 2)))


def g(x):
    return (x - 20) / pow(2, 2) * f(x)


wave_equation_solver = WaveEquationSolver(n, m, delta_t, delta_x)
wave_equation_solver.solve_explicit(f, g)

wave_equation_solver.draw_n(0)
wave_equation_solver.draw_n(n)
