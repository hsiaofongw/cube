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

image = matrix_renderer.render(
    camera, 
    pixels, 
    points_a, 
    points_b, 
    points_c
)

image = image.reshape(img_height, img_width)

plt.imshow(image)
plt.savefig('cube.png')