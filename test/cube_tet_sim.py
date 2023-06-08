import taichi as ti
import numpy as np

from interface import render_funcs, mesh_render_3d
from data import cage_data, tet_data, lbs_data
from cons import deform3d, framework
import compdyn.base, compdyn.inverse
from utils import objs

import time

ti.init(arch=ti.x64, cpu_max_num_threads=1)

# ========================== load data ==========================
modelname = 'cube_tet'
tgf_path = f'assets/{modelname}/{modelname}_cage.tgf'
model_path = f'assets/{modelname}/{modelname}.mesh'
weight_path = f'assets/{modelname}/{modelname}_w.txt'
scale = 1.0
repose = (0.0, 0.0, 0.0)

cage = cage_data.load_cage_data(tgf_path, weight_path, scale, repose)
mesh = tet_data.load_tets(model_path, scale, repose)
lbs = lbs_data.LBS3D(v_p_ref=mesh.v_p_ref,
                     c_p=cage.c_p,
                     c_p_ref=cage.c_p_ref,
                     v_weights=cage.v_weights)
wireframe = [True]

# ========================== init simulation ==========================
g = ti.Vector([0.0, 0.0, 0.0])
fps = 60
substeps = 3
subsub = 1
dt = 1.0 / fps / substeps

pbd = framework.pbd_framework(mesh.v_p, g, dt, damp=0.9993)
deform = deform3d.Deform3D(v_p=mesh.v_p,
                           v_p_ref=mesh.v_p_ref,
                           v_invm=mesh.v_invm,
                           t_i=mesh.t_i,
                           t_m=mesh.t_m,
                           dt=dt,
                           hydro_alpha=1e-2,
                           devia_alpha=1e-2)
comp = compdyn.base.CompDynBase(v_p=mesh.v_p,
                                v_p_ref=mesh.v_p_ref,
                                v_p_rig=lbs.v_p_rig,
                                v_invm=mesh.v_invm,
                                c_p=cage.c_p,
                                c_p_ref=cage.c_p_ref,
                                v_weights=cage.v_weights,
                                dt=dt,
                                alpha=1e-5)
comp_inv = compdyn.inverse.CompDynInv(v_p=mesh.v_p,
                                      v_p_ref=mesh.v_p_ref,
                                      v_p_rig=lbs.v_p_rig,
                                      v_invm=mesh.v_invm,
                                      c_p=cage.c_p,
                                      c_p_ref=cage.c_p_ref,
                                      v_weights=cage.v_weights,
                                      dt=dt,
                                      alpha=1e-4)
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
window.add_render_func(render_func=render_funcs.get_mesh_render_func(
    lbs.v_p_rig, mesh.f_i, wireframe, color=(0.0, 1.0, 0.0)))
window.add_render_func(
    render_funcs.get_cage_render_func(cage, point_radius=0.04, edge_width=1.0))

# ========================== init status ==========================
pbd.init_rest_status(0)

# ========================== use input ==========================
import math


def set_movement():
  t = window.get_time()
  p = cage.points_np
  p_input = cage.c_p.to_numpy()
  p_input[7] = p[7] + np.array([1.0, 0.0, -1.0], dtype=np.float32) * math.cos(
      0.5 * t * (2.0 * math.pi)) * 0.5
  p_input[2] = p[2] + np.array([-1.0, 0.0, 1.0], dtype=np.float32) * math.cos(
      0.5 * t * (2.0 * math.pi)) * 0.5
  cage.c_p.from_numpy(p_input)


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
    pbd.update_cons(1)
    t_total += time.time() - t
    lbs.linear_blend_skinning()
    pbd.update_vel()

  window.pre_update()
  window.render()
  window.show()

window.terminate()

print(t_total / window.get_total_frames(), window.get_total_frames())

# ========================== plot data ==========================
verts = mesh.v_p.to_numpy()
faces = mesh.f_i.to_numpy().reshape((-1, 3))

import meshio

meshio.write('cube_tet.obj',
             meshio.Mesh(points=verts, cells=[("triangle", faces)]))

import pyvista

mesh = pyvista.read('cube_tet.obj')
mesh.plot(smooth_shading=True, show_edges=True)