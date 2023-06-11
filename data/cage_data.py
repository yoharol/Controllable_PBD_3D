import taichi as ti
import numpy as np
from utils import io


class CageData:

  def __init__(self,
               points: np.ndarray,
               weights: np.ndarray,
               edges: np.ndarray,
               scale=1.0,
               repose=(0.0, 0.0, 0.0)) -> None:
    assert edges.ndim == 1, 'edges should be 1D array'

    self.n_points = points.shape[0]
    self.n_edges = edges.shape[0] // 2

    points = points * scale + np.array(repose, dtype=np.float32)
    self.points_np = points
    self.weights_np = weights
    self.edges_np = edges

    self.c_p = ti.Vector.field(3, dtype=ti.f32, shape=self.n_points)
    self.c_p_ref = ti.Vector.field(3, dtype=ti.f32, shape=self.n_points)
    self.c_p_input = ti.Vector.field(3, dtype=ti.f32, shape=self.n_points)
    self.v_weights = ti.field(dtype=ti.f32, shape=weights.shape)
    self.e_i = ti.field(dtype=ti.i32, shape=self.n_edges * 2)

    self.c_p.from_numpy(points)
    self.c_p_ref.from_numpy(points)
    self.c_p_input.from_numpy(points)
    self.v_weights.from_numpy(weights)
    self.e_i.from_numpy(edges)

    self.c_color = ti.Vector.field(3, dtype=ti.f32, shape=self.n_points)

  def set_color(self,
                point_color=(0.0, 1.0, 0.0),
                fixed_color=(1.0, 0.0, 0.0),
                fixed=[]):
    color = self.c_color.to_numpy()
    color[:] = np.array(point_color)
    color[fixed] = np.array(fixed_color)
    self.c_color.from_numpy(color)


def load_cage_data(tgfpath: str,
                   weightpath: str,
                   scale=1.0,
                   repose=(0.0, 0.0, 0.0)):
  points, bone_edges, cage_edges = io.tgf_loader(tgfpath)
  if bone_edges.shape[0] != 0:
    assert False, 'bone_edges should be empty'
  weights = np.loadtxt(weightpath)
  return CageData(points, weights, cage_edges, scale, repose)
