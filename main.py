from render import SimpleRenderer
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

renderer = SimpleRenderer()
image = renderer.render(
    camera, 
    pixels, 
    points_a, 
    points_b, 
    points_c, 
    img_width, 
    img_height
)

plt.imshow(image)
plt.savefig('cube.png')