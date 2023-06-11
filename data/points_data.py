import taichi as ti
import numpy as np
from utils import io


class PointsData:

  def __init__(self,
               points: np.ndarray,
               weights: np.ndarray,
               scale=1.0,
               repose=(0.0, 0.0, 0.0)) -> None:

    self.n_points = points.shape[0]

    points = points * scale + np.array(repose, dtype=np.float32)
    self.points_np = points
    self.weights_np = weights

    self.c_p = ti.Vector.field(3, dtype=ti.f32, shape=self.n_points)
    self.c_p_ref = ti.Vector.field(3, dtype=ti.f32, shape=self.n_points)
    self.c_p_input = ti.Vector.field(3, dtype=ti.f32, shape=self.n_points)
    self.v_weights = ti.field(dtype=ti.f32, shape=weights.shape)

    self.c_A = ti.Matrix.field(3, 3, dtype=ti.f32, shape=self.n_points)
    self.c_A.from_numpy(
        np.eye(3, dtype=np.float32).repeat(self.n_points, 0).reshape(-1, 3, 3))
    self.c_b = ti.Vector.field(3, dtype=ti.f32, shape=self.n_points)
    self.c_b.from_numpy(np.zeros((self.n_points, 3), dtype=np.float32))

    self.c_p.from_numpy(points)
    self.c_p_ref.from_numpy(points)
    self.c_p_input.from_numpy(points)

    self.v_weights.from_numpy(weights)

    self.c_color = ti.Vector.field(3, dtype=ti.f32, shape=self.n_points)

  def set_color(self,
                point_color=(0.0, 1.0, 0.0),
                fixed_color=(1.0, 0.0, 0.0),
                fixed=[]):
    color = self.c_color.to_numpy()
    color[:] = np.array(point_color)
    color[fixed] = np.array(fixed_color)
    self.c_color.from_numpy(color)


def load_points_data(tgfpath: str,
                     weightpath: str,
                     scale=1.0,
                     repose=(0.0, 0.0, 0.0)):
  points, bone_edges, cage_edges = io.tgf_loader(tgfpath)
  if cage_edges.shape[0] != 0:
    assert False, 'cage_edges should be empty'
  if bone_edges.shape[0] != 0:
    assert False, 'bone_edges should be empty'
  weights = np.loadtxt(weightpath)
  return PointsData(points, weights, scale, repose)
