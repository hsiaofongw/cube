from canvas import Canvas
from cube import Cube
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

cube_center = np.array([12, 0, 0])
edge_length = 4
cube = Cube(cube_center, edge_length)
cube.rotate(
    np.array([0, 0, 1]),
    cube.cube_center,
    np.radians(15)
)

cube_vertices = cube.cube_vertices
ax.scatter(
    cube_vertices[:, 0],
    cube_vertices[:, 1],
    cube_vertices[:, 2]
)

print(cube.get_edge_lengths())

camera = np.array([0, 0, 2])

canvas_distance = 6

view_direction = cube_center - camera

canvas = Canvas(
    camera,
    view_direction,
    canvas_distance
)

horizon = np.array([0, -1, 0])
canvas.set_horizon(horizon)

angles_h = np.radians(np.array([-30, 30]))
angles_v = np.radians(np.array([-30, 30]))
img_width = 20
img_height = 15
pixels = canvas.get_pixel_points(
    angles_h,
    angles_v,
    img_width,
    img_height
)

camera = np.atleast_2d(camera)
view_directions = pixels - camera

np.savetxt(
    'views.txt',
    view_directions,
    '%.2f'
)

ax.set_xlim(0, 20)
ax.set_ylim(-10, 10)
ax.set_zlim(0, 20)

plt.xlabel('xlabel')
plt.ylabel('ylabel')

ax.quiver(
    camera[:, 0:1],
    camera[:, 1:2],
    camera[:, 2:3],
    view_directions[:, 0], 
    view_directions[:, 1], 
    view_directions[:, 2]
)

plt.show()