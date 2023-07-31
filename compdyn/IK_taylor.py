import taichi as ti
import numpy as np
from scipy import linalg

import utils.mathlib


@ti.data_oriented
class PointsTaylorIK:

  def __init__(self,
               v_p: ti.MatrixField,
               v_p_ref: ti.MatrixField,
               v_weights: ti.Field,
               v_invm: ti.Field,
               c_p: ti.MatrixField,
               c_p_ref: ti.MatrixField,
               c_p_input: ti.MatrixField,
               trans_idx=[]) -> None:
    assert len(trans_idx) > 0

    self.n_verts = v_p.shape[0]
    self.n_points = c_p.shape[0]
    self.n_free_trans_points = len(trans_idx)

    self.v_p = v_p
    self.v_p_ref = v_p_ref
    self.v_weights = v_weights
    self.v_invm = v_invm
    self.c_p = c_p
    self.c_p_ref = c_p_ref
    self.c_p_input = c_p_input

    self.c_R = ti.Matrix.field(3, 3, dtype=ti.f32, shape=self.n_points)
    self.c_t = ti.Vector.field(3, dtype=ti.f32, shape=self.n_free_trans_points)
    self.c_rotvec = ti.Vector.field(3, dtype=ti.f32, shape=self.n_points)
    self.v_p_rig = ti.Vector.field(3, dtype=ti.f32, shape=self.n_verts)
    # self.rot_mat = ti.Matrix.field(3, 3, dtype=ti.f32, shape=(self.n_points, self.n_verts))

    self.trans_idx = trans_idx
    self.t_idx = ti.field(dtype=ti.i32, shape=self.n_free_trans_points)
    self.t_idx.from_numpy(np.array(self.trans_idx))
    #self.corres_idx = ti.field(dtype=ti.i32, shape=self.n_points)
    #for i in range(self.n_free_trans_points):

    self.init_data()

  def init_data(self):
    self.init_ti_data()

    self.solver_dim = 3 * self.n_free_trans_points + 3 * self.n_points
    self.LHS = np.zeros((self.solver_dim, self.solver_dim), dtype=np.float32)
    #self.M = np.zeros((self.n_verts, self.n_verts), dtype=np.float32)
    #for i in range(self.n_verts):
    #  self.M[i,i] = 1.0 / self.v_invm[i]
    self.Wt = np.zeros((3 * self.n_verts, 3 * self.n_free_trans_points),
                       dtype=np.float32)
    for tidx in range(self.n_free_trans_points):
      j = self.trans_idx[tidx]
      for i in range(self.n_verts):
        self.Wt[i * 3, tidx * 3] = self.v_weights[i, j]
        self.Wt[i * 3 + 1, tidx * 3 + 1] = self.v_weights[i, j]
        self.Wt[i * 3 + 2, tidx * 3 + 2] = self.v_weights[i, j]

    self.WtM = np.zeros_like(self.Wt)
    for tidx in range(self.n_free_trans_points):
      j = self.trans_idx[tidx]
      for i in range(self.n_verts):
        self.WtM[i * 3, tidx * 3] = self.v_weights[i, j] / self.v_invm[i]
        self.WtM[i * 3 + 1,
                 tidx * 3 + 1] = self.v_weights[i, j] / self.v_invm[i]
        self.WtM[i * 3 + 2,
                 tidx * 3 + 2] = self.v_weights[i, j] / self.v_invm[i]

    self.Wr = np.zeros(dtype=np.float32,
                       shape=(3 * self.n_verts, 3 * self.n_points))
    self.MWr = np.zeros(dtype=np.float32,
                        shape=(3 * self.n_verts, 3 * self.n_points))
    self.Wr_ti = ti.field(dtype=ti.f32,
                          shape=(self.n_verts * 3, self.n_points * 3))
    self.MWr_ti = ti.field(dtype=ti.f32,
                           shape=(self.n_verts * 3, self.n_points * 3))

    self.update_rot_solver()

    # self.LHS[:] = self.WtM.T @ self.Wt
    # ! update LHS

  def update_rot_solver(self):
    self.update_ti_rot_solver()
    # self.Wr[:] = self.Wr_ti.to_numpy().reshape(self.Wr.shape)
    self.Wr = self.Wr_ti.to_numpy()
    self.MWr = self.MWr_ti.to_numpy()
    self.LHS = np.bmat([[self.WtM.T @ self.Wt, self.WtM.T @ self.Wr],
                        [self.Wr.T @ self.WtM, self.MWr.T @ self.Wr]])

  @ti.kernel
  def update_ti_rot_solver(self):
    for i in range(self.n_verts):
      for j in range(self.n_points):
        vec = self.v_weights[i, j] * self.c_R[j] @ (self.v_p_ref[i] -
                                                    self.c_p_ref[j])
        mat = -utils.mathlib.get_cross_matrix(vec)
        for k in ti.static(range(3)):
          for l in ti.static(range(3)):
            self.Wr_ti[i * 3 + k, j * 3 + l] = mat[k, l]
            self.MWr_ti[i * 3 + k, j * 3 + l] = mat[k, l] / self.v_invm[i]

  def ik(self):
    self.linear_blend_skinning()

    self.update_rot_solver()
    diff_vp = (self.v_p.to_numpy() - self.v_p_rig.to_numpy()).flatten()
    RHS = np.concatenate((self.WtM.T @ diff_vp, self.MWr.T @ diff_vp))

    delta_u = linalg.solve(self.LHS, RHS, assume_a='sym')
    np.savetxt("test.txt", self.LHS, fmt="%.3f")
    delta_u = delta_u.reshape(-1, 3)
    self.c_t.from_numpy(delta_u[:self.n_free_trans_points] +
                        self.c_t.to_numpy())
    self.c_rotvec.from_numpy(delta_u[self.n_free_trans_points:])
    self.update_ti_data()
    self.linear_blend_skinning()

  @ti.kernel
  def init_ti_data(self):
    for j in range(self.n_points):
      self.c_R[j] = ti.Matrix.identity(ti.f32, 3)
    for idxj in range(self.n_free_trans_points):
      j = self.t_idx[idxj]
      self.c_t[idxj] = self.c_p[j]

  @ti.kernel
  def update_ti_data(self):
    for j in range(self.n_free_trans_points):
      self.c_p[self.t_idx[j]] = self.c_t[j]
    for j in range(self.n_points):
      vec = self.c_rotvec[j]
      self.c_R[j] = utils.mathlib.rotvec_to_matrix(vec.normalized(),
                                                   vec.norm()) @ self.c_R[j]

  @ti.kernel
  def linear_blend_skinning(self):
    for i in range(self.n_verts):
      self.v_p_rig[i] = ti.Vector([0.0, 0.0, 0.0])
      for j in range(self.n_points):
        self.v_p_rig[i] += self.v_weights[i, j] * (
            self.c_R[j] @ (self.v_p_ref[i] - self.c_p_ref[j]) + self.c_p[j])
