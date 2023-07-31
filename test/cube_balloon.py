import taichi as ti
import numpy as np

from interface import render_funcs, mesh_render_3d
from data import cage_data, cloth_data, lbs_data
from cons import length, bend, volume, framework
import compdyn.base, compdyn.inverse, compdyn.IK
from interface import usd_objs, usd_render
from utils import objs

import time

ti.init(arch=ti.x64, cpu_max_num_threads=1)

# ========================== load data ==========================
modelname = 'cube_mesh'
tgf_path = f'assets/{modelname}/{modelname}_cage.tgf'
model_path = f'assets/{modelname}/{modelname}.obj'
weight_path = f'assets/{modelname}/{modelname}_w.txt'
scale = 1.0
repose = (0.0, 0.0, 0.0)

fixed = [0, 1, 4, 2, 5, 7]

cage = cage_data.load_cage_data(tgf_path, weight_path, scale, repose)
cage.set_color(fixed=fixed)
mesh = cloth_data.load_cloth_mesh(model_path, scale, repose)
lbs = lbs_data.CageLBS3D(v_p_ref=mesh.v_p_ref,
                         c_p=cage.c_p,
                         c_p_ref=cage.c_p_ref,
                         v_weights=cage.v_weights)
wireframe = [False]
wireframe2 = [True]

# ========================== init simulation ==========================
g = ti.Vector([0.0, 0.0, 0.0])
fps = 60
substeps = 5
subsub = 1
dt = 1.0 / fps / substeps

pbd = framework.pbd_framework(mesh.v_p, g, dt, damp=0.993)
cons_length = length.LengthCons(v_p=mesh.v_p,
                                v_p_ref=mesh.v_p_ref,
                                e_i=mesh.e_i,
                                e_rl=mesh.e_rl,
                                v_invm=mesh.v_invm,
                                dt=dt,
                                alpha=1e-2)
cons_bend = bend.Bend3D(v_p=mesh.v_p,
                        v_p_ref=mesh.v_p_ref,
                        e_i=mesh.e_i,
                        e_s=mesh.e_s,
                        v_invm=mesh.v_invm,
                        dt=dt,
                        alpha=100)
cons_volume = volume.Volume(v_p=mesh.v_p,
                            v_p_ref=mesh.v_p_ref,
                            f_i=mesh.f_i,
                            v_invm=mesh.v_invm,
                            dt=dt,
                            alpha=0.0)
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
pbd.add_cons(cons_length, 0)
pbd.add_cons(cons_bend, 0)
pbd.add_cons(cons_volume, 0)
pbd.add_cons(comp, 1)

ground = objs.Quad(axis1=(10.0, 0.0, 0.0), axis2=(0.0, 0.0, -10.0), pos=0.0)
pbd.add_collision(ground.collision)

# ========================== init interface ==========================
window = mesh_render_3d.MeshRender3D(res=(1000, 1000),
                                     title='cube_balloon',
                                     kernel='taichi')
window.set_background_color((1, 1, 1, 1))
# window.add_render_func(ground.get_render_draw())
window.add_render_func(
    render_funcs.get_mesh_render_func(mesh.v_p,
                                      mesh.f_i,
                                      wireframe,
                                      color=(14 / 255, 87 / 255, 204 / 255)))
"""window.add_render_func(render_func=render_funcs.get_mesh_render_func(
    lbs.v_p_rig, mesh.f_i, wireframe, color=(0.0, 1.0, 0.0)))"""
window.add_render_func(
    render_funcs.get_cage_render_func(cage,
                                      point_radius=0.04,
                                      edge_width=8.0,
                                      edge_color=(255 / 255, 145 / 255,
                                                  0 / 255)))

# ========================== init status ==========================
pbd.init_rest_status(0)

# ========================== user input ==========================
import math

written = [False]


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
    meshio.Mesh(mesh.v_p.to_numpy(), [("triangle", mesh.faces_np.reshape(-1, 3))
                                     ]).write("out/cube_balloon.obj")
    print("write to outputs/cube_balloon.obj")
    print(cage.c_p.to_numpy())
    written[0] = True


# ========================== USD ==========================
save_usd = True
if save_usd:
  stage = usd_render.UsdRender('out/cube_balloon.usdc',
                               startTimeCode=1,
                               endTimeCode=600,
                               fps=60,
                               UpAxis='Y')
  cage_point_color = np.zeros((cage.n_points, 3), dtype=np.float32)
  cage_point_color[:] = np.array([0.0, 1.0, 0.0], dtype=np.float32)
  cage_point_color[comp.fixed] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
  usd_cage_points = usd_objs.SavePoints(stage.stage,
                                        '/root/cage_points',
                                        verts_np=cage.c_p.to_numpy(),
                                        radius=0.05,
                                        per_vert_color=cage_point_color)
  usd_cage_edges = usd_objs.SaveLines(stage.stage,
                                      '/root/cage_edges',
                                      verts_np=cage.c_p.to_numpy(),
                                      edges_np=cage.edges_np,
                                      width=0.025,
                                      color=(0.0, 1.0, 1.0))
  usd_mesh = usd_objs.SaveMesh(stage.stage, '/root/mesh', mesh.verts_np,
                               mesh.faces_np)

  def update_usd(frame: int):
    if frame < stage.startTimeCode or frame > stage.endTimeCode:
      return
    usd_cage_points.update(cage.c_p.to_numpy(), frame)
    usd_cage_edges.update(cage.c_p.to_numpy(), frame)
    usd_mesh.update(mesh.v_p.to_numpy(), frame)
    print("update usd file at frame", frame)


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