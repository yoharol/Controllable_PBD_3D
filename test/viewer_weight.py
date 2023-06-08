import taichi as ti
import numpy as np
import meshio

from utils import mathlib

ti.init(arch=ti.gpu)

modelname = 'cube_mesh'
meshpath = f'assets/{modelname}/{modelname}.obj'
cagepath = f'assets/{modelname}/{modelname}_cage.tgf'
weightpath = f'assets/{modelname}/{modelname}_w.txt'

mesh = meshio.read(meshpath)
points = mesh.points
faces = mesh.cells[0].data
weight = np.loadtxt(weightpath)
print(f'points: {points.shape}, faces: {faces.shape},  weight: {weight.shape}')

n_verts = points.shape[0]
v_p = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)
v_color = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)
w = ti.field(dtype=ti.f32, shape=weight.shape)

v_p.from_numpy(points)
v_color.fill(ti.Vector([0.0, 0.0, 0.0]))
w.from_numpy(weight)


@ti.kernel
def set_color(index: ti.i32):
  for i in range(n_verts):
    v_color[i] = mathlib.heat_rgb(w[i, index], 0.0, 1.0)


window = ti.ui.Window('Weights Viewer', (800, 800), vsync=True)
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.Camera()
camera.position(4.0, 4.0, 4.0)
camera.lookat(0.5, 0.5, 0.5)
camera.up(0, 1, 0)
camera.projection_mode(ti.ui.ProjectionMode.Perspective)
scene.set_camera(camera)
gui = window.get_gui()

index = 0
set_color(index)

while window.running:
  if window.get_event(ti.ui.PRESS):
    if window.event.key == 'z':
      index -= 1
    elif window.event.key == 'x':
      index += 1
    index = index % weight.shape[1]
    set_color(index)

  camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
  scene.set_camera(camera)
  scene.ambient_light((0.8, 0.8, 0.8))
  scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))

  scene.particles(v_p, radius=0.01, per_vertex_color=v_color)
  canvas.scene(scene)

  with gui.sub_window("Sub Window", x=0, y=0, width=0.2, height=0.2):
    gui.text(f"{index}")

  window.show()
