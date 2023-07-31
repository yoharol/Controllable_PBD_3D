import taichi as ti
import numpy as np

from utils import geom2d, geom3d
from utils import io


class ClothData:

  def __init__(self,
               verts: np.ndarray,
               faces: np.ndarray,
               scale=1.0,
               repose=(0.0, 0.0, 0.0)) -> None:
    self.n_verts = verts.shape[0]
    self.n_faces = faces.shape[0] // 3

    assert faces.ndim == 1, 'faces should be 1D array'

    verts = verts * scale + np.array(repose, dtype=np.float32)
    self.verts_np = verts
    self.faces_np = faces

    self.v_p = ti.Vector.field(3, dtype=ti.f32, shape=self.n_verts)
    self.v_p_ref = ti.Vector.field(3, dtype=ti.f32, shape=self.n_verts)
    self.f_i = ti.field(dtype=ti.i32, shape=self.n_faces * 3)
    self.v_p.from_numpy(verts)
    self.v_p_ref.from_numpy(verts)
    self.f_i.from_numpy(faces)

    edge_indices, edge_sides, edge_neib, face_edges = geom2d.edge_extractor(
        faces)
    self.e_i = ti.field(dtype=ti.i32, shape=edge_indices.shape[0])
    self.e_s = ti.field(dtype=ti.i32, shape=edge_sides.shape[0])
    self.e_n = ti.field(dtype=ti.i32, shape=edge_neib.shape[0])
    self.f_e = ti.field(dtype=ti.i32, shape=face_edges.shape[0])

    self.e_i.from_numpy(edge_indices)
    self.e_s.from_numpy(edge_sides)
    self.e_n.from_numpy(edge_neib)
    self.f_e.from_numpy(face_edges)

    rest_length = geom2d.compute_rest_length(verts, edge_indices)
    self.e_rl = ti.field(dtype=ti.f32, shape=edge_indices.shape[0] // 2)
    self.e_rl.from_numpy(rest_length)

    vert_mass, face_mass = geom2d.compute_vert_mass(verts, faces)
    self.v_invm = ti.field(dtype=ti.f32, shape=self.n_verts)
    self.f_m = ti.field(dtype=ti.f32, shape=self.n_faces)
    self.v_invm.from_numpy(1.0 / vert_mass)
    self.f_m.from_numpy(face_mass)


def load_cloth_mesh(meshpath: str,
                    scale=1.0,
                    repose=(0.0, 0.0, 0.0),
                    remove_duplicate=True,
                    reverse_side=False):
  assert meshpath.endswith('.obj'), 'Only support .obj file'
  verts, faces = io.load_obj(meshpath)
  if remove_duplicate:
    faces = faces.reshape((-1, 3))
    faces = faces[::2]
    faces = faces.reshape((-1))
  if reverse_side:
    geom3d.revert_all_faces(faces)
  return ClothData(verts, faces, scale, repose)
