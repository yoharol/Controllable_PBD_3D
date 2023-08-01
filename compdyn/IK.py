import taichi as ti
import numpy as np
from scipy import linalg

from data import cage_data


@ti.data_oriented
class CageIK:

  def __init__(self,
               v_p: ti.MatrixField,
               v_p_ref: ti.MatrixField,
               v_weights: ti.Field,
               v_invm: ti.Field,
               c_p: ti.MatrixField,
               c_p_ref: ti.MatrixField,
               c_p_input: ti.MatrixField,
               fix_trans=[]) -> None:
    self.n_verts = v_p.shape[0]
    self.n_points = c_p.shape[0]

    self.v_p = v_p
    self.v_p_ref = v_p_ref
    self.verts_ref = v_p_ref.to_numpy()
    self.v_weights = v_weights
    self.v_invm = v_invm
    self.c_p = c_p
    self.c_p_ref = c_p_ref
    self.c_p_input = c_p_input
    self.n_fixed = 0

    self.c_b = ti.Vector.field(3, dtype=ti.f32, shape=self.n_points)

    self.init_data()
    self.set_fixedpoint(fix_trans)

  def init_data(self):
    self.W = np.zeros(dtype=np.float32,
                      shape=(self.n_verts * 3, self.n_points * 3))
    self.WTM = np.zeros(dtype=np.float32,
                        shape=(self.n_points * 3, self.n_verts * 3))

    for i in range(self.n_verts):
      for j in range(self.n_points):
        w = self.v_weights[i, j]
        wm = w * self.v_invm[i]
        self.W[i * 3, j * 3] = w
        self.W[i * 3 + 1, j * 3 + 1] = w
        self.W[i * 3 + 2, j * 3 + 2] = w

        self.WTM[j * 3, i * 3] = wm
        self.WTM[j * 3 + 1, i * 3 + 1] = wm
        self.WTM[j * 3 + 2, i * 3 + 2] = wm
    self.WTMW = self.WTM @ self.W
    self.LHS = self.WTMW

  def set_fixedpoint(self, fix_trans):
    if len(fix_trans) > 0:
      self.n_fixed = len(fix_trans)
      self.fixed_idx = fix_trans
      cons_list = []
      for ft in fix_trans:
        g = np.zeros(dtype=np.float32, shape=(3, self.n_points * 3))
        g[0, ft * 3] = 1.0
        g[1, ft * 3 + 1] = 1.0
        g[2, ft * 3 + 2] = 1.0
        cons_list.append(g)
      G = np.vstack(cons_list)
      self.LHS = np.bmat([[self.WTMW, G.T], [G, np.zeros_like((G @ G.T))]])
    self.solver = linalg.inv(self.LHS)

  def ik(self):
    RHS = self.WTM @ (self.v_p.to_numpy() - self.verts_ref).flatten()
    if self.n_fixed > 0:
      bc = np.zeros((self.n_fixed * 3), dtype=np.float32)
      for i in range(self.n_fixed):
        idx = self.fixed_idx[i]
        bc[i * 3] = self.c_p_input[idx][0]
        bc[i * 3 + 1] = self.c_p_input[idx][1]
        bc[i * 3 + 2] = self.c_p_input[idx][2]
      RHS = np.concatenate([RHS, bc])
    P = self.solver @ RHS
    self.c_p.from_numpy(P[:self.n_points * 3].reshape(self.n_points, 3))
    self.update_after_solver()

  @ti.kernel
  def update_after_solver(self):
    for i in range(self.n_points):
      self.c_p[i] = self.c_p[i] + self.c_p_ref[i]


@ti.data_oriented
class PointsIK:

  def __init__(self,
               v_p: ti.MatrixField,
               v_p_ref: ti.MatrixField,
               v_weights: ti.Field,
               v_invm: ti.Field,
               c_p: ti.MatrixField,
               c_p_ref: ti.MatrixField,
               c_p_input: ti.MatrixField,
               fix_trans=[]) -> None:
    self.n_verts = v_p.shape[0]
    self.n_points = c_p.shape[0]

    self.v_p = v_p
    self.v_p_ref = v_p_ref
    self.v_weights = v_weights
    self.v_invm = v_invm
    self.c_p = c_p
    self.c_p_ref = c_p_ref
    self.c_p_ref_np = c_p_ref.to_numpy()
    self.c_p_input = c_p_input
    self.n_fixed = 0

    self.c_b = ti.Vector.field(3, dtype=ti.f32, shape=self.n_points)
    self.c_rot = ti.Matrix.field(3, 3, dtype=ti.f32, shape=self.n_points)
    self.M = np.zeros(dtype=np.float32, shape=(self.n_verts, self.n_verts))
    self.W = np.zeros(dtype=np.float32, shape=(self.n_verts, self.n_points * 4))
    self.T = np.zeros(dtype=np.float32, shape=(self.n_points * 4, 3))
    self.T_ti = ti.field(dtype=ti.f32, shape=(self.n_points * 4, 3))
    self.X_hat = np.zeros(dtype=np.float32, shape=(self.n_verts, 3))
    self.c_p_ref_W = np.zeros(dtype=np.float32,
                              shape=(self.n_points, self.n_points * 4))

    self.v_p_rig = ti.Vector.field(3, dtype=ti.f32, shape=self.n_verts)

    self.init_data()
    self.set_fixedpoint(fix_trans)

  def init_data(self):
    for i in range(self.n_verts):
      self.M[i, i] = 1.0 / self.v_invm[i]
    for i in range(self.n_verts):
      for j in range(self.n_points):
        w = self.v_weights[i, j]
        self.W[i, j * 4] = w * (self.v_p_ref[i][0] - self.c_p_ref[j][0])
        self.W[i, j * 4 + 1] = w * (self.v_p_ref[i][1] - self.c_p_ref[j][1])
        self.W[i, j * 4 + 2] = w * (self.v_p_ref[i][2] - self.c_p_ref[j][2])
        self.W[i, j * 4 + 3] = w
        self.X_hat[i] += w * self.c_p_ref_np[j]
    self.LHS = self.W.T @ self.M @ self.W
    self.WTM = self.W.T @ self.M

    c_p_ref_np = self.c_p_ref.to_numpy()
    for j in range(self.n_points):
      self.c_p_ref_W[j, j * 4:j * 4 + 3] = 0.0
      self.c_p_ref_W[j, j * 4 + 3] = 1.0

  def set_fixedpoint(self, fix_trans):
    self.n_fixed = len(fix_trans)
    self.fix_trans = fix_trans
    if self.n_fixed > 0:
      cons_list = []
      for ft in fix_trans:
        G = np.zeros(dtype=np.float32, shape=(1, self.n_points * 4))
        G[0, ft * 4] = 0.0
        G[0, ft * 4 + 1] = 0.0
        G[0, ft * 4 + 2] = 0.0
        G[0, ft * 4 + 3] = 1.0
        cons_list.append(G)
      G = np.vstack(cons_list)
      self.LHS = np.bmat([[self.LHS, G.T], [G, np.zeros_like(G @ G.T)]])
      assert self.n_fixed == G.shape[0]
    self.LHS_inv = linalg.inv(self.LHS)

  def ik(self):
    RHS = self.WTM @ (self.v_p.to_numpy() - self.X_hat)
    if self.n_fixed > 0:
      RHS = np.vstack([RHS, np.zeros((self.n_fixed, 3))])
      n_ft = len(self.fix_trans)
      for ft_idx in range(n_ft):
        idx = self.fix_trans[ft_idx]
        RHS[self.n_points * 4 + ft_idx,
            0] = self.c_p_input[idx][0] - self.c_p_ref[idx][0]
        RHS[self.n_points * 4 + ft_idx,
            1] = self.c_p_input[idx][1] - self.c_p_ref[idx][1]
        RHS[self.n_points * 4 + ft_idx,
            2] = self.c_p_input[idx][2] - self.c_p_ref[idx][2]
    t = self.LHS_inv @ RHS
    t = t[:self.n_points * 4, :]
    self.T_ti.from_numpy(t)
    self.polar_decompose(t)
    self.v_p_rig.from_numpy(self.W @ t + self.X_hat)
    self.c_p.from_numpy(self.c_p_ref_W @ t + self.c_p_ref_np)

  def polar_decompose(self, t: np.ndarray):
    rot = np.zeros(dtype=np.float32, shape=(self.n_points, 3, 3))
    trans = np.zeros(dtype=np.float32, shape=(self.n_points, 3))
    for j in range(self.n_points):
      A = t[j * 4:(j + 1) * 4 - 1, :].T
      R, S = linalg.polar(A)
      rot[j] = R
      trans[j] = t[(j + 1) * 4 - 1, :]
      self.T[j * 4:(j + 1) * 4 - 1, :] = R.T
    self.c_b.from_numpy(trans)
    self.c_rot.from_numpy(rot)

  @ti.kernel
  def lbs(self):
    for i in range(self.n_verts):
      self.v_p_rig[i] = ti.Vector.zero(ti.f32, 3)
      for j in range(self.n_points):
        self.v_p_rig[i] += self.v_weights[i, j] * (
            self.c_rot[j] @ (self.v_p[i] - self.c_p_ref[j]) + self.c_p[j])
