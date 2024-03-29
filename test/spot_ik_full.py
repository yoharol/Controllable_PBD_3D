import taichi as ti
import numpy as np

from interface import render_funcs, mesh_render_3d
from data import tet_data, points_data
from cons import deform3d, framework
import compdyn.base, compdyn.inverse, compdyn.IK, compdyn.point
from utils import objs
from interface import usd_objs, usd_render

import time

ti.init(arch=ti.x64, cpu_max_num_threads=1)

# ========================== load data ==========================
modelname = 'spot'
tgf_path = f'assets/{modelname}/{modelname}.tgf'
model_path = f'assets/{modelname}/{modelname}.mesh'
weight_path = f'assets/{modelname}/{modelname}_w.txt'
scale = 1.0
repose = (0.0, 1.7, 0.0)

points = points_data.load_points_data(tgf_path, weight_path, scale, repose)
mesh = tet_data.load_tets(model_path, scale, repose)
np.savetxt("test.txt", mesh.faces_np.reshape(-1, 3), fmt='%d')
wireframe = [False]
fixed = []
points.set_color(fixed=fixed)

# ========================== init simulation ==========================
g = ti.Vector([0.0, -3.0, 0.0])
fps = 60
substeps = 5
subsub = 1
dt = 1.0 / fps / substeps

pbd = framework.pbd_framework(mesh.v_p, g, dt, damp=1.0 - 1e-3)
deform = deform3d.Deform3D(v_p=mesh.v_p,
                           v_p_ref=mesh.v_p_ref,
                           v_invm=mesh.v_invm,
                           t_i=mesh.t_i,
                           t_m=mesh.t_m,
                           dt=dt,
                           hydro_alpha=1e-2,
                           devia_alpha=5e-2)
points_ik = compdyn.IK.PointsIK(v_p=mesh.v_p,
                                v_p_ref=mesh.v_p_ref,
                                v_weights=points.v_weights,
                                v_invm=mesh.v_invm,
                                c_p=points.c_p,
                                c_p_ref=points.c_p_ref,
                                c_p_input=points.c_p_input,
                                fix_trans=fixed)
"""comp = compdyn.base.CompDynBase(v_p=mesh.v_p,
                                v_p_ref=mesh.v_p_ref,
                                v_p_rig=points_ik.v_p_rig,
                                v_invm=mesh.v_invm,
                                c_p=points.c_p,
                                c_p_ref=points.c_p_ref,
                                v_weights=points.v_weights,
                                dt=dt,
                                alpha=3e-5,
                                alpha_fixed=1e-5,
                                fixed=fixed)"""
comp = compdyn.point.CompDynPoint(v_p=mesh.v_p,
                                  v_p_ref=mesh.v_p_ref,
                                  v_p_rig=points_ik.v_p_rig,
                                  v_invm=mesh.v_invm,
                                  c_p=points.c_p,
                                  c_p_ref=points.c_p_ref,
                                  c_rot=points_ik.c_rot,
                                  v_weights=points.v_weights,
                                  dt=dt,
                                  alpha=1e-5,
                                  alpha_fixed=1e-5,
                                  fixed=fixed)
pbd.add_cons(deform, 0)
pbd.add_cons(comp, 1)

ground = objs.Quad(axis1=(10.0, 0.0, 0.0), axis2=(0.0, 0.0, -10.0), pos=0.0)
pbd.add_collision(ground.collision)

# ========================== init interface ==========================
window = mesh_render_3d.MeshRender3D(res=(700, 700),
                                     title='cube_tet',
                                     kernel='taichi')
window.add_render_func(ground.get_render_draw())
window.add_render_func(
    render_funcs.get_mesh_render_func(mesh.v_p,
                                      mesh.f_i,
                                      wireframe,
                                      color=(1.0, 1.0, 1.0)))
# window.add_render_func(render_func=render_funcs.get_mesh_render_func(
#     points_ik.v_p_rig, mesh.f_i, wireframe, color=(0.0, 0.0, 1.0)))
window.add_render_func(
    render_funcs.get_points_render_func(points, point_radius=0.04))

# ========================== init status ==========================
pbd.init_rest_status(0)
pbd.init_rest_status(1)

# ========================== use input ==========================
import math


def set_movement():
  t = window.get_time() - 1.0
  p_input = points_ik.c_p_ref.to_numpy()
  if t > 0.0:
    idx1 = 7
    p_input[idx1] += np.array([1.0, 1.0, -1.0], dtype=np.float32) * math.sin(
        0.5 * t * (2.0 * math.pi)) * 0.25
  #p_input[2] = p[2] + np.array([-1.0, 0.0, 1.0], dtype=np.float32) * math.sin(
  #    0.5 * t * (2.0 * math.pi)) * 0.5
  points.c_p_input.from_numpy(p_input)


# ========================== usd rneder ==========================
save_usd = False
if save_usd:
  stage = usd_render.UsdRender('out/spot_free.usdc',
                               startTimeCode=1,
                               endTimeCode=480,
                               fps=60,
                               UpAxis='Y')
  point_color = np.zeros((points.n_points, 3), dtype=np.float32)
  point_color[:] = np.array([0.0, 1.0, 0.0], dtype=np.float32)
  point_color[comp.fixed] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
  usd_cage_points = usd_objs.SavePoints(stage.stage,
                                        '/root/spot_points',
                                        verts_np=points.c_p.to_numpy(),
                                        radius=0.05,
                                        per_vert_color=point_color)
  mesh.update_surface_verts()
  usd_mesh = usd_objs.SaveMesh(stage.stage, '/root/mesh',
                               mesh.surface_v_p.to_numpy(),
                               mesh.surface_f_i.to_numpy())

  def update_usd(frame: int):
    if frame < stage.startTimeCode or frame > stage.endTimeCode:
      print("update_usd: frame out of range")
      return
    usd_cage_points.update(points.c_p.to_numpy(), frame)
    mesh.update_surface_verts()
    usd_mesh.update(mesh.surface_v_p.to_numpy(), frame)
    print("update usd file at frame", frame)


t_total = 0.0

while window.running():

  set_movement()

  for i in range(substeps):
    pbd.make_prediction()
    pbd.preupdate_cons(0)
    pbd.preupdate_cons(1)
    for j in range(subsub):
      pbd.update_cons(0)
    t = time.time()
    points_ik.ik()
    pbd.update_cons(1)
    t_total += time.time() - t
    pbd.update_vel()

  if save_usd:
    update_usd(window.get_total_frames())

  window.pre_update()
  window.render()
  window.show()

window.terminate()

if save_usd:
  stage.save()