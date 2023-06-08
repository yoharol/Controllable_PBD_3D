import taichi as ti
import numpy as np
from scipy import linalg


@ti.data_oriented
class CompDynInv:

  def __init__(self,
               v_p: ti.MatrixField,
               v_p_ref: ti.MatrixField,
               v_p_rig: ti.MatrixField,
               v_invm: ti.Field,
               c_p: ti.MatrixField,
               c_p_ref: ti.MatrixField,
               v_weights: ti.Field,
               dt,
               alpha=0.0) -> None:
    self.n_verts = v_p.shape[0]
    self.n_points = c_p.shape[0]

    self.v_p = v_p
    self.v_p_ref = v_p_ref
    self.v_p_rig = v_p_rig
    self.v_invm = v_invm
    self.c_p = c_p
    self.c_p_ref = c_p_ref
    self.v_weights = v_weights
    self.dt = dt
    self.alpha = alpha / dt / dt

    self.W_ti = ti.field(dtype=ti.f32,
                         shape=(self.n_points * 3, self.n_verts * 3))
    self.WMinv_ti = ti.field(dtype=ti.f32,
                             shape=(self.n_points * 3, self.n_verts * 3))
    self.delta_lambda_ti = ti.field(dtype=ti.f32, shape=(self.n_points * 3))
    self.flambda = np.zeros(dtype=np.float32, shape=self.n_points * 3)

    self.C = ti.field(dtype=ti.f32, shape=self.n_points * 3)

    self.init_data()

  def init_data(self):
    self.init_ti_data()
    self.W = self.W_ti.to_numpy()
    self.WMinv = self.WMinv_ti.to_numpy()
    self.WMinv_T = self.WMinv.T
    self.solver = linalg.inv(self.WMinv @ self.W.T + self.alpha)

  def preupdate_cons(self):
    self.flambda.fill(0.0)

  def update_cons(self):
    x = self.v_p.to_numpy().flatten()
    x_rig = self.v_p_rig.to_numpy().flatten()
    C = self.W @ (x - x_rig)
    delta_lambda = -self.solver @ (C + self.alpha * self.flambda)
    x += self.WMinv_T @ delta_lambda
    self.flambda += delta_lambda
    self.v_p.from_numpy(x.reshape(self.n_verts, 3))

  def test(self):
    x = self.v_p.to_numpy().flatten()
    x_rig = self.v_p_rig.to_numpy().flatten()
    return self.W @ (x - x_rig)

  @ti.kernel
  def comupte_C(self):
    self.C.fill(0.0)
    for i in range(self.n_verts):
      x_c = self.v_p[i] - self.v_p_rig[i]
      for j in range(self.n_points):
        m = 1.0 / self.v_invm[i]
        w = self.v_weights[i, j]
        self.C[j * 3 + 0] += m * w * x_c[0]
        self.C[j * 3 + 1] += m * w * x_c[1]
        self.C[j * 3 + 2] += m * w * x_c[2]

  @ti.kernel
  def init_ti_data(self):
    for i in range(self.n_verts):
      for j in range(self.n_points):
        for k in ti.static(range(3)):
          self.W_ti[j * 3 + k,
                    i * 3 + k] = self.v_weights[i, j] / self.v_invm[i]
          self.WMinv_ti[j * 3 + k, i * 3 + k] = self.v_weights[i, j]
