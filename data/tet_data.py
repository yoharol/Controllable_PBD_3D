import taichi as ti
import numpy as np

from utils import geom3d
from utils import io


@ti.data_oriented
class TetData:

  def __init__(self,
               verts: np.ndarray,
               faces: np.ndarray,
               tets: np.ndarray,
               scale=1.0,
               repose=(0.0, 0.0, 0.0)) -> None:
    assert faces.ndim == 1, 'faces should be 1D array'
    assert tets.ndim == 1, 'tets should be 1D array'

    self.n_verts = verts.shape[0]
    self.n_faces = faces.shape[0] // 3
    self.n_tets = tets.shape[0] // 4

    verts = verts * scale + np.array(repose, dtype=np.float32)

    self.verts_np = verts
    self.faces_np = faces

    self.v_p = ti.Vector.field(3, dtype=ti.f32, shape=self.n_verts)
    self.v_p_ref = ti.Vector.field(3, dtype=ti.f32, shape=self.n_verts)
    self.f_i = ti.field(dtype=ti.i32, shape=self.n_faces * 3)
    self.t_i = ti.field(dtype=ti.i32, shape=self.n_tets * 4)

    self.v_p.from_numpy(verts)
    self.v_p_ref.from_numpy(verts)
    self.f_i.from_numpy(faces)
    self.t_i.from_numpy(tets)

    vert_order, vert_mass, tet_mass = geom3d.compute_vertex_mass(verts, tets)

    self.v_invm = ti.field(dtype=ti.f32, shape=self.n_verts)
    self.t_m = ti.field(dtype=ti.f32, shape=self.n_tets)
    self.v_invm.from_numpy(1.0 / vert_mass)
    self.t_m.from_numpy(tet_mass)
    self.extract_surface()

  def extract_surface(self):
    vert_idx = []
    corres_idx = np.zeros(self.n_verts, dtype=np.int32)
    corres_idx[:] = -1
    for i in range(self.n_faces):
      for j in range(3):
        idx = self.faces_np[i * 3 + j]
        if idx not in vert_idx:
          vert_idx.append(idx)
          corres_idx[idx] = len(vert_idx) - 1

    face_idx = np.zeros_like(self.faces_np)
    for i in range(self.n_faces):
      for j in range(3):
        face_idx[i * 3 + j] = corres_idx[self.faces_np[i * 3 + j]]
        assert face_idx[i * 3 + j] != -1

    self.n_surface_verts = len(vert_idx)
    self.surface_v_p = ti.Vector.field(3,
                                       dtype=ti.f32,
                                       shape=self.n_surface_verts)
    self.surface_v_i = ti.field(dtype=ti.i32, shape=self.n_surface_verts)
    self.surface_v_i.from_numpy(np.array(vert_idx, dtype=np.int32))
    self.surface_f_i = ti.field(dtype=ti.i32, shape=self.n_faces * 3)
    self.surface_f_i.from_numpy(face_idx)

    self.update_surface_verts()

  @ti.kernel
  def update_surface_verts(self):
    for i in range(self.n_surface_verts):
      self.surface_v_p[i] = self.v_p[self.surface_v_i[i]]


def load_tets(tetpath: str,
              scale=1.0,
              repose=(0.0, 0.0, 0.0),
              reverse_face=False,
              remove_duplicate=True):
  assert tetpath.endswith('.mesh'), 'tetpath should be .mesh file'
  verts, tets, faces = io.load_tet(tetpath, reverse_face)
  if remove_duplicate:
    faces = faces.reshape((-1, 3))
    faces = faces[::2]
    faces = faces.reshape((-1))

  faces = geom3d.revert_all_faces(faces)
  return TetData(verts, faces, tets, scale, repose)
