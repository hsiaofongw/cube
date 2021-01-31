from render import SimpleRenderer, CPUMatrixRenderer
import matplotlib.pyplot as plt
from scenarios.s1 import Scenario1

print("正在布置场景……", end=" ")
scene1 = Scenario1()
canvas, triangles = scene1.get()
print("Done.", end="\n")

print("正在加载渲染器……", end=" ")
simple_renderer = SimpleRenderer()
matrix_renderer = CPUMatrixRenderer()
print("Done.", end="\n")

print("开始渲染……", end="\n")
image1 = matrix_renderer.render(canvas, triangles)
print("Done.", end="\n")

print("开始整合计算结果……", end=" ")
image1 = image1.reshape(canvas.height_pixels, canvas.width_pixels)
print("Done.", end="\n")

print("将图片存盘……", end=" ")
plt.imsave('image1.png', image1)
print("Done.", end="\n")
