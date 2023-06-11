import taichi as ti
import numpy as np

from interface import render_funcs, mesh_render_3d
from data import cage_data, tet_data, lbs_data
from cons import deform3d, framework
import compdyn.base, compdyn.inverse, compdyn.IK
from utils import objs
from interface import usd_objs, usd_render

import time

ti.init(arch=ti.x64, cpu_max_num_threads=1)

# ========================== load data ==========================
modelname = 'cube_tet'
tgf_path = f'assets/{modelname}/{modelname}_cage.tgf'
model_path = f'assets/{modelname}/{modelname}.mesh'
weight_path = f'assets/{modelname}/{modelname}_w.txt'
scale = 1.0
repose = (0.0, 0.0, 0.0)

fixed = [0, 1, 4, 2, 5, 7]
cage = cage_data.load_cage_data(tgf_path, weight_path, scale, repose)
cage.set_color(fixed=fixed)
mesh = tet_data.load_tets(model_path, scale, repose)
lbs = lbs_data.CageLBS3D(v_p_ref=mesh.v_p_ref,
                         c_p=cage.c_p,
                         c_p_ref=cage.c_p_ref,
                         v_weights=cage.v_weights)
wireframe = [True]

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
comp = compdyn.base.CompDynBase(v_p=mesh.v_p,
                                v_p_ref=mesh.v_p_ref,
                                v_p_rig=lbs.v_p_rig,
                                v_invm=mesh.v_invm,
                                c_p=cage.c_p,
                                c_p_ref=cage.c_p_ref,
                                v_weights=cage.v_weights,
                                dt=dt,
                                alpha=1e-3,
                                alpha_fixed=1e-4,
                                fixed=fixed)
cage_ik = compdyn.IK.CageIK(v_p=mesh.v_p,
                            v_p_ref=mesh.v_p_ref,
                            v_weights=cage.v_weights,
                            v_invm=mesh.v_invm,
                            c_p=cage.c_p,
                            c_p_ref=cage.c_p_ref,
                            c_p_input=cage.c_p_input,
                            fix_trans=fixed)
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
"""window.add_render_func(render_func=render_funcs.get_mesh_render_func(
    lbs.v_p_rig, mesh.f_i, wireframe, color=(0.0, 1.0, 0.0)))"""
window.add_render_func(
    render_funcs.get_cage_render_func(cage, point_radius=0.04, edge_width=1.0))

# ========================== init status ==========================
pbd.init_rest_status(0)

# ========================== use input ==========================
import math

written = [True]


def set_movement():
  t = window.get_time() - 1.0
  p = cage.points_np
  p_input = np.zeros((cage_ik.n_points, 3), dtype=np.float32)
  if t > 0.0:
    p_input[7] = np.array([1.0, 0.3, -1.0], dtype=np.float32) * math.sin(
        0.5 * t * (2.0 * math.pi)) * 0.35
    p_input[2] = np.array([-1.0, 0.3, 1.0], dtype=np.float32) * math.sin(
        0.5 * t * (2.0 * math.pi)) * 0.35
  cage.c_p_input.from_numpy(p_input)
  if abs(0.5 * t - 1.25) < 1e-2 and not written[0]:
    import meshio
    meshio.Mesh(
        mesh.v_p.to_numpy(),
        [("triangle", mesh.faces_np.reshape(-1, 3))]).write("out/cube_tet.obj")
    print("write to outputs/cube_tet.obj")
    print(cage.c_p.to_numpy())
    written[0] = True


# ========================== USD ==========================
save_usd = False
if save_usd:
  stage = usd_render.UsdRender('out/cube_tet.usdc',
                               startTimeCode=1,
                               endTimeCode=240,
                               fps=60,
                               UpAxis='Y')
  cage_point_color = np.zeros((len(fixed), 3), dtype=np.float32)
  cage_point_color[:] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
  usd_cage_points = usd_objs.SavePoints(stage.stage,
                                        '/root/cage_points',
                                        verts_np=mesh.v_p.to_numpy()[fixed],
                                        radius=0.05,
                                        per_vert_color=cage_point_color)
  usd_mesh = usd_objs.SaveMesh(stage.stage, '/root/mesh', mesh.verts_np,
                               mesh.faces_np)

  def update_usd(frame: int):
    if frame < stage.startTimeCode or frame > stage.endTimeCode:
      return
    usd_cage_points.update(mesh.v_p.to_numpy()[fixed], frame)
    usd_mesh.update(mesh.v_p.to_numpy(), frame)


t_total = 0.0

while window.running():

  t = time.time()
  set_movement()

  for i in range(substeps):
    pbd.make_prediction()
    pbd.preupdate_cons(0)
    pbd.preupdate_cons(1)
    for j in range(subsub):
      pbd.update_cons(0)
    cage_ik.ik()
    pbd.update_cons(1)
    lbs.linear_blend_skinning()
    pbd.update_vel()
  t_total += time.time() - t
  if window.get_total_frames() == 480:
    print(f'average time: {t_total / 480}')

  if save_usd:
    update_usd(window.get_total_frames())

  window.pre_update()
  window.render()
  window.show()

window.terminate()
if save_usd:
  stage.save()