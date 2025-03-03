import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class WaveEquationSolver:
    def __init__(self, n, m, delta_t, delta_x):
        self.N = n
        self.M = m
        self.delta_t = delta_t
        self.delta_x = delta_x
        self.courant_param = self.delta_t / self.delta_x
        self.grid = np.empty([self.N + 1, self.M + 1])

    def second_centered_spatial_derivative(self, n):
        return self.grid[n, 2:] - 2 * self.grid[n, 1:-1] + self.grid[n, :-2]

    def second_centered_time_derivative(self, n):
        return self.grid[n + 1, 1:-1] - 2 * self.grid[n, 1:-1] + self.grid[n - 1, 1:-1]

    def spatial_domain(self):
        return np.linspace(0, self.M * self.delta_x, self.M + 1)

    def time_domain(self):
        return np.linspace(0, self.N * self.delta_t, self.N + 1)

    def solve_explicit(self, f, g):
        # boundary conditions
        self.grid[:, 0] = self.grid[:, self.M] = 0  # Dirichlet boundary condition (fixed ends)

        # initial conditions
        self.grid[0, 1:-1] = f(self.spatial_domain()[1:-1])
        self.grid[1, 1:-1] = (self.grid[0, 1:-1] + self.delta_t * g(self.spatial_domain()[1:-1])
                              + self.courant_param ** 2 / 2 * self.second_centered_spatial_derivative(0))

        # solver (explicit)
        for n in range(1, self.N):
            self.grid[n + 1, 1:-1] = (2 * self.grid[n, 1:-1] - self.grid[n - 1, 1:-1]
                                      + self.courant_param ** 2 * self.second_centered_spatial_derivative(n))

    def solution_n(self, n):
        return self.grid[n, :]

    def solution_m(self, m):
        return self.grid[:, m]

    def solution_mn(self, n, m):
        return self.grid[n, m]

    def draw_n(self, n):
        plt.plot(self.spatial_domain(), self.solution_n(n))
        plt.ylim(1.2 * -np.max(self.grid), 1.2 * np.max(self.grid))
        plt.show()

    def draw_m(self, m):
        plt.plot(self.time_domain(), self.solution_m(m))
        plt.show()

    def animate(self):
        fig, ax = plt.subplots()
        ax.set_ylim(1.2 * -np.max(self.grid), 1.2 * np.max(self.grid))
        line, = ax.plot(self.spatial_domain(), self.solution_n(0))

        def update(frame):
            line.set_ydata(self.solution_n(frame))
            return line,

        anim = animation.FuncAnimation(fig, update, frames=self.N, blit=True)
        writer = animation.FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        anim.save("animation.mp4", writer=writer)
