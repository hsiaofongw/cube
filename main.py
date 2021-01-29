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

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

cube.rotate(
    np.array([0, 0, 1]),
    cube.cube_center,
    pi/6
)

cube.move(np.array([6, 0, -2]))

vertices1 = cube.cube_vertices.copy()

# ax.set_zlim(-1, 15)
# ax.set_xlim(-1, 15)
# ax.set_ylim(-1, 15)

# plt.xlabel('xlabel')
# plt.ylabel('ylabel')

# ax.scatter(vertices0[:, 0], vertices0[:, 1], vertices0[:, 2], marker='o')
# ax.scatter(vertices1[:, 0], vertices1[:, 1], vertices1[:, 2], marker='^')

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

view_vectors = pixels - camera
camera = np.reshape(camera, newshape=(1,3,))

# ax.scatter(pixels[:, 0], pixels[:, 1], pixels[:, 2], marker='s')

# ax.quiver(
#     camera[:, 0], 
#     camera[:, 1], 
#     camera[:, 2], 
#     view_vectors[:, 0],
#     view_vectors[:, 1],
#     view_vectors[:, 2]
# )

points_a, points_b, points_c = cube.get_facades()
vecs_ab = points_b - points_a
vecs_ac = points_c - points_a
triangles_norm = np.cross(vecs_ab, vecs_ac, axisa=1, axisb=1)
triangles_center = (points_a + points_b + points_c) / 3

# ax.quiver(
#     triangles_center[:, 0],
#     triangles_center[:, 1],
#     triangles_center[:, 2],
#     triangles_norm[:, 0],
#     triangles_norm[:, 1],
#     triangles_norm[:, 2]
# )

cosines_views_triangles = Helper.cosine_many_to_many(
    view_vectors,
    triangles_norm
)

criterion = 0.3

# np.savetxt('cosines.csv', cosines_views_triangles, fmt='%.2f', delimiter=', ')

# plt.show()


x = np.abs(cosines_views_triangles)
selectors = x >= criterion

# the histogram of the data
# n, bins, patches = plt.hist(x, 50, density=True)

# plt.grid(True)
# plt.show()


# np.savetxt('selectors.csv', selectors, fmt='%1d', delimiter=', ')

n_triangles = points_a.shape[0]
n_pixels = view_vectors.shape[0]

camera = camera.reshape(1, 3)
print(camera.shape)

print(points_a.shape)

print(view_vectors.shape)

print(vecs_ab.shape)

print(vecs_ac.shape)

camera = np.reshape(camera, newshape=(1,3,))

points_a = np.repeat(
    points_a,
    n_pixels,
    axis = 1
).flatten(order='F')
points_a = points_a.reshape(points_a.shape[0], 1)

view_vectors = np.repeat(
    view_vectors,
    n_triangles,
    axis=0
).flatten(order='C')

view_vectors = np.reshape(
    view_vectors,
    newshape=(view_vectors.shape[0], 1,)
)

vecs_ab = np.repeat(
    vecs_ab,
    n_pixels,
    axis=1
).flatten(order='F')
vecs_ab = vecs_ab.reshape(vecs_ab.shape[0], 1)

vecs_ac = np.repeat(
    vecs_ac,
    n_pixels,
    axis=1
).flatten(order='F')
vecs_ac = vecs_ac.reshape(vecs_ac.shape[0], 1)

coeff_B = camera - points_a

print(coeff_B.shape)

coeff_A = np.concatenate(
    (-view_vectors, vecs_ab, vecs_ac,),
    axis=1
)

print(coeff_A.shape)