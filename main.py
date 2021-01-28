import numpy as np
from math import pi

from cube import Cube
from helper import Helper
from canvas import Canvas

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


cube_center = np.array([6, 0, 0])
edge_length = 1
cube = Cube(
    cube_center,
    edge_length
)

vertices0 = cube.cube_vertices.copy()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

cube.rotate(
    np.array([0, 0, 1]),
    cube.cube_center,
    pi/6
)

cube.move(np.array([6, 0, -2]))

vertices1 = cube.cube_vertices.copy()

ax.set_zlim(-1, 20)
ax.set_xlim(-1, 20)
ax.set_ylim(-1, 20)

plt.xlabel('xlabel')
plt.ylabel('ylabel')

# ax.scatter(vertices0[:, 0], vertices0[:, 1], vertices0[:, 2], marker='o')
ax.scatter(vertices1[:, 0], vertices1[:, 1], vertices1[:, 2], marker='^')

camera = np.array([0, 0, 1])
view_direction = cube.cube_center - camera

canvas = Canvas(
    camera,
    view_direction,
    4
)

horizon = np.array([0, -1, 0])
canvas.set_horizon(horizon)

pixels = canvas.get_pixel_points(
    np.array([-pi/4, pi/4]), 
    np.array([-pi/6, pi/6]),
    20, 10
)

ax.scatter(pixels[:, 0], pixels[:, 1], pixels[:, 2], marker='s')

plt.show()
