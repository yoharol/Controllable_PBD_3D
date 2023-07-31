import taichi as ti
import numpy as np

from interface import render_funcs, mesh_render_3d
from data import tet_data
from cons import framework, deform3d

from interface import usd_objs, usd_render
from utils import objs

import time

ti.init(arch=ti.x64, cpu_max_num_threads=1)

# ========================== load data ==========================
modelname = 'cube_tet'
tgf_path = f'assets/{modelname}/{modelname}_cage.tgf'
model_path = f'assets/{modelname}/{modelname}2.mesh'
weight_path = f'assets/{modelname}/{modelname}_w.txt'
scale = 1.0
repose = (0.0, 0.0, 0.0)

mesh = tet_data.load_tets(model_path, scale, repose)

# ========================== set fixed points ==========================

fixed = [0, 1, 4, 5, 2, 7]
corres = [0, 1, 2, 3, 4, 5, 6, 7]
invm = mesh.v_invm.to_numpy()
invm[fixed] = 0.0
mesh.v_invm.from_numpy(invm)

display_points = ti.Vector.field(3, dtype=ti.f32, shape=len(fixed))


def update_display_points():
  for i in range(len(fixed)):
    for k in range(3):
      display_points[i][k] = mesh.v_p[fixed[i]][k]


wireframe = [False]

# ========================== init simulation ==========================
g = ti.Vector([0.0, 0.0, 0.0])
fps = 60
substeps = 5
subsub = 1
dt = 1.0 / fps / substeps

pbd = framework.pbd_framework(mesh.v_p, g, dt, damp=0.993)
deform = deform3d.Deform3D(v_p=mesh.v_p,
                           v_p_ref=mesh.v_p_ref,
                           v_invm=mesh.v_invm,
                           t_i=mesh.t_i,
                           t_m=mesh.t_m,
                           dt=dt,
                           hydro_alpha=1e-2,
                           devia_alpha=5e-2)
pbd.add_cons(deform, 0)

ground = objs.Quad(axis1=(10.0, 0.0, 0.0), axis2=(0.0, 0.0, -10.0), pos=0.0)
pbd.add_collision(ground.collision)

# ========================== init interface ==========================
window = mesh_render_3d.MeshRender3D(res=(1000, 1000),
                                     title='cube_tet',
                                     kernel='taichi')
#window.add_render_func(ground.get_render_draw())
window.add_render_func(
    render_funcs.get_mesh_render_func(mesh.surface_v_p,
                                      mesh.surface_f_i,
                                      wireframe,
                                      color=(14 / 255, 87 / 255, 204 / 255)))
window.add_render_func(
    render_funcs.get_particles_render_func(display_points,
                                           color=(1.0, 0.0, 0.0),
                                           radius=0.04))

# ========================== init status ==========================
pbd.init_rest_status(0)

# ========================== use input ==========================
import math

written = [False]


def set_movement():
  t = window.get_time() - 1.0
  if t < 0.0:
    return
  v_p = mesh.v_p.to_numpy()
  v_p_ref = mesh.v_p_ref.to_numpy()

  v_p[7] = v_p_ref[7] + np.array([1.0, 0.3, -1.0], dtype=np.float32) * math.sin(
      0.5 * t * (2.0 * math.pi)) * 0.35
  v_p[2] = v_p_ref[2] + np.array([-1.0, 0.3, 1.0], dtype=np.float32) * math.sin(
      0.5 * t * (2.0 * math.pi)) * 0.35

  if abs(0.5 * t - 1.25) < 1e-2 and not written[0]:
    import meshio
    meshio.Mesh(v_p, [("triangle", mesh.faces_np.reshape(-1, 3))
                     ]).write("out/cube_tet_drag.obj")
    print("write to outputs/cube_tet_drag.obj")
    print(v_p[corres])
    written[0] = True
  mesh.v_p.from_numpy(v_p)


# ========================== USD ==========================
save_usd = True
if save_usd:
  stage = usd_render.UsdRender('out/cube_tet_drag.usdc',
                               startTimeCode=1,
                               endTimeCode=600,
                               fps=60,
                               UpAxis='Y')
  cage_point_color = np.zeros((len(fixed), 3), dtype=np.float32)
  cage_point_color[:] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
  usd_cage_points = usd_objs.SavePoints(stage.stage,
                                        '/root/cage_points',
                                        verts_np=mesh.v_p.to_numpy()[fixed],
                                        radius=0.05,
                                        per_vert_color=cage_point_color)
  mesh.update_surface_verts()
  usd_mesh = usd_objs.SaveMesh(stage.stage, '/root/mesh',
                               mesh.surface_v_p.to_numpy(),
                               mesh.surface_f_i.to_numpy())

  def update_usd(frame: int):
    if frame < stage.startTimeCode or frame > stage.endTimeCode:
      return
    usd_cage_points.update(mesh.v_p.to_numpy()[fixed], frame)
    mesh.update_surface_verts()
    usd_mesh.update(mesh.surface_v_p.to_numpy(), frame)
    print("update usd file at frame", frame)


t_total = 0.0

while window.running():

  t = time.time()
  set_movement()
  for i in range(substeps):
    pbd.make_prediction()
    pbd.preupdate_cons(0)
    for j in range(subsub):
      pbd.update_cons(0)
    pbd.update_vel()
  t_total += time.time() - t
  if window.get_total_frames() == 480:
    print(f'average time: {t_total / 480}')

  mesh.update_surface_verts()
  update_display_points()

  if save_usd:
    update_usd(window.get_total_frames())

  window.pre_update()
  window.render()
  window.show()

window.terminate()
if save_usd:
  stage.save()