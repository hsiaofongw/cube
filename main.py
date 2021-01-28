import numpy as np
from math import pi

from cube import Cube
from helper import Helper
from canvas import Canvas

cube_center = np.array([6, 0, 0])
edge_length = 1
cube = Cube(
    cube_center,
    edge_length
)
np.savetxt('vertexes0.csv', cube.cube_vertices, delimiter=',')

cube.rotate(
    np.array([0, 0, 1]),
    cube.cube_center,
    pi/6
)
np.savetxt('vertexes1.csv', cube.cube_vertices, delimiter=',')
