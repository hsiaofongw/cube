from render import SimpleRenderer, CPUMatrixRenderer
import matplotlib.pyplot as plt
from scenarios.s1 import Scenario1

scene1 = Scenario1()

camera, \
view_directions, \
points_a, \
points_b, \
points_c, \
pixels, \
img_width, \
img_height = scene1.get()

simple_renderer = SimpleRenderer()
matrix_renderer = CPUMatrixRenderer()

image1 = matrix_renderer.render(
    camera, 
    pixels, 
    points_a, 
    points_b, 
    points_c
)
image1 = image1.reshape(img_height, img_width)
plt.imsave('image1.png', image1)

# image2 = simple_renderer.render(
#     camera,
#     pixels,
#     points_a,
#     points_b,
#     points_c
# )
# image2 = image2.reshape(img_height, img_width)
# plt.imsave('image2.png', image2)