import taichi as ti
import numpy as np
from scipy import linalg

from data import cage_data


class CageIK:

  def __init__(self,
               v_p: ti.MatrixField,
               v_weights: ti.Field,
               v_invm: ti.Field,
               c_p: ti.MatrixField,
               c_p_ref: ti.MatrixField,
               c_p_input: ti.MatrixField,
               fix_trans=[]) -> None:
    self.n_verts = v_p.shape[0]
    self.n_points = c_p.shape[0]

    self.v_p = v_p
    self.v_weights = v_weights
    self.v_invm = v_invm
    self.c_p = c_p
    self.c_p_ref = c_p_ref
    self.c_p_input = c_p_input
    self.n_fixed = 0

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
    RHS = self.WTM @ self.v_p.to_numpy().flatten()
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
