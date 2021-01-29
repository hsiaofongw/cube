from canvas import Canvas
from cube import Cube
import numpy as np
from helper import Helper

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

cube_center = np.array([10, 0, 0])
edge_length = 3
cube = Cube(cube_center, edge_length)
cube.rotate(
    np.array([0, 0, 1]),
    cube.cube_center,
    np.radians(15)
)

cube_vertices = cube.cube_vertices
# ax.scatter(
#     cube_vertices[:, 0],
#     cube_vertices[:, 1],
#     cube_vertices[:, 2]
# )

camera = np.array([0, 2, 4])

canvas_distance = 3

view_direction = cube_center - camera

canvas = Canvas(
    camera,
    view_direction,
    canvas_distance
)

horizon = np.array([0, -1, 0])
canvas.set_horizon(horizon)

angles_h = np.radians(np.array([-30, 30]))
angles_v = np.radians(np.array([-20, 20]))
img_width = 300
img_height = 200
pixels = canvas.get_pixel_points(
    angles_h,
    angles_v,
    img_width,
    img_height
)

camera = np.atleast_2d(camera)
view_directions = pixels - camera

# ax.set_xlim(0, 20)
# ax.set_ylim(-10, 10)
# ax.set_zlim(0, 20)

# plt.xlabel('xlabel')
# plt.ylabel('ylabel')

# ax.quiver(
#     camera[:, 0:1],
#     camera[:, 1:2],
#     camera[:, 2:3],
#     view_directions[:, 0], 
#     view_directions[:, 1], 
#     view_directions[:, 2]
# )

# plt.show()

image = np.zeros(
    shape = (img_height, img_width,), 
    dtype = np.int
)

points_a, points_b, points_c = cube.get_facades()
vecs_ab = points_b - points_a
vecs_ac = points_c - points_a
triangle_norms = np.cross(
    vecs_ab,
    vecs_ac,
    axisa=1,
    axisb=1
)

cosines = np.abs(Helper.cosine_many_to_many(
    view_directions,
    triangle_norms
))

criterion = 1e-10

camera = camera.T

for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        k = i * img_width + j
        pixel = 0
        for l in range(points_a.shape[0]):
            print(f"i: {i}, j: {j}, k: {k}, l: {l}")

            cosine = cosines[k, l]
            if cosine < criterion:
                continue

            v = view_directions[k:(k+1), :].T
            point_a = points_a[l:(l+1), :].T
            vec_ab = vecs_ab[l:(l+1), :].T
            vec_ac = vecs_ac[l:(l+1), :].T

            coeff_A = np.concatenate(
                (-v, vec_ab, vec_ac,),
                axis=1
            )

            coeff_B = camera - point_a

            sol = np.linalg.solve(coeff_A, coeff_B)
            z = sol[0, 0]
            x = sol[1, 0]
            y = sol[2, 0]

            if (x + y <= 1) and (x >= 0) and (y >= 0):
                pixel = pixel + 1
                break
        
        if pixel >= 1:
            image[i, j] = 1

plt.imshow(image)
plt.savefig('cube.png')