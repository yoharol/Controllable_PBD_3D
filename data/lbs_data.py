import taichi as ti
import numpy as np


@ti.data_oriented
class CageLBS3D:

  def __init__(self, v_p_ref: ti.MatrixField, c_p: ti.MatrixField,
               c_p_ref: ti.MatrixField, v_weights: ti.Field) -> None:
    self.n_verts = v_p_ref.shape[0]
    self.n_points = c_p.shape[0]

    assert v_weights.shape == (self.n_verts, self.n_points)
    self.v_p_ref = v_p_ref
    self.c_p = c_p
    self.c_p_ref = c_p_ref
    self.v_weights = v_weights

    self.v_p_rig = ti.Vector.field(3, ti.f32, shape=self.n_verts)
    self.v_p_rig.copy_from(self.v_p_ref)

  @ti.kernel
  def linear_blend_skinning(self):
    for i in range(self.n_verts):
      self.v_p_rig[i] = ti.Vector([0.0, 0.0, 0.0])
      for j in range(self.n_points):
        self.v_p_rig[i] += self.v_weights[i, j] * (
            self.v_p_ref[i] + self.c_p[j] - self.c_p_ref[j])


@ti.data_oriented
class LBS3D:

  def __init__(self, v_p_ref: ti.MatrixField, c_A: ti.MatrixField,
               c_b: ti.MatrixField, v_weights: ti.Field) -> None:
    self.n_verts = v_p_ref.shape[0]
    self.n_points = c_b.shape[0]

    assert v_weights.shape == (self.n_verts, self.n_points)
    self.v_p_ref = v_p_ref
    self.c_b = c_b
    self.c_A = c_A
    self.v_weights = v_weights

    self.v_p_rig = ti.Vector.field(3, ti.f32, shape=self.n_verts)
    self.v_p_rig.copy_from(self.v_p_ref)

  @ti.kernel
  def linear_blend_skinning(self):
    for i in range(self.n_verts):
      self.v_p_rig[i] = ti.Vector([0.0, 0.0, 0.0])
      for j in range(self.n_points):
        self.v_p_rig[i] += self.v_weights[i, j] * (
            self.c_A[j] @ self.v_p_ref[i] + self.c_b[j])
