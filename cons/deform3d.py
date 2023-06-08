import taichi as ti
from utils.mathlib import *


@ti.data_oriented
class Deform3D:

  def __init__(self,
               v_p: ti.MatrixField,
               v_p_ref: ti.MatrixField,
               v_invm: ti.Field,
               t_i: ti.Field,
               t_m: ti.Field,
               dt: float,
               hydro_alpha=0.0,
               devia_alpha=0.0) -> None:
    self.n = t_i.shape[0] // 4
    self.hydro_lambda = ti.field(dtype=ti.f32, shape=self.n)
    self.devia_lambda = ti.field(dtype=ti.f32, shape=self.n)
    self.hydro_alpha = hydro_alpha
    self.devia_alpha = devia_alpha
    self.t_i = t_i
    self.v_invm = v_invm
    self.t_m = t_m
    self.v_p = v_p
    self.v_p_ref = v_p_ref
    self.dt = dt

  def init_rest_status(self):
    pass

  def preupdate_cons(self):
    self.devia_lambda.fill(0.0)
    self.hydro_lambda.fill(0.0)

  def update_cons(self):
    self.solve_cons()

  @ti.kernel
  def solve_cons(self):
    for k in range(self.n):
      a = self.t_i[k * 4]
      b = self.t_i[k * 4 + 1]
      c = self.t_i[k * 4 + 2]
      d = self.t_i[k * 4 + 3]
      x_1 = self.v_p[a]
      x_2 = self.v_p[b]
      x_3 = self.v_p[c]
      x_4 = self.v_p[d]
      r_1 = self.v_p_ref[a]
      r_2 = self.v_p_ref[b]
      r_3 = self.v_p_ref[c]
      r_4 = self.v_p_ref[d]
      w1 = self.v_invm[a]
      w2 = self.v_invm[b]
      w3 = self.v_invm[c]
      w4 = self.v_invm[d]
      D = ti.Matrix.cols([x_1 - x_4, x_2 - x_4, x_3 - x_4])
      B = ti.Matrix.cols([r_1 - r_4, r_2 - r_4, r_3 - r_4]).inverse()
      F = D @ B
      f1 = ti.Vector([F[0, 0], F[1, 0], F[2, 0]])
      f2 = ti.Vector([F[0, 1], F[1, 1], F[2, 1]])
      f3 = ti.Vector([F[0, 2], F[1, 2], F[2, 2]])

      C_H = F.determinant() - 1
      par_det_F = ti.Matrix.cols([f2.cross(f3), f3.cross(f1), f1.cross(f2)])
      CH_H = par_det_F @ (B.transpose())
      par_CH_x1 = ti.Vector([CH_H[0, 0], CH_H[1, 0], CH_H[2, 0]])
      par_CH_x2 = ti.Vector([CH_H[0, 1], CH_H[1, 1], CH_H[2, 1]])
      par_CH_x3 = ti.Vector([CH_H[0, 2], CH_H[1, 2], CH_H[2, 2]])
      par_CH_x4 = -par_CH_x1 - par_CH_x2 - par_CH_x3
      sum_par_CH = w1 * par_CH_x1.norm_sqr() + w2 * par_CH_x2.norm_sqr(
      ) + w3 * par_CH_x3.norm_sqr() + w4 * par_CH_x4.norm_sqr()
      alpha_tilde_H = self.hydro_alpha / (self.dt * self.dt * self.t_m[k])

      C_D = F.norm_sqr() - 3.0
      CD_H = 2.0 * F @ (B.transpose())
      par_CD_x1 = ti.Vector([CD_H[0, 0], CD_H[1, 0], CD_H[2, 0]])
      par_CD_x2 = ti.Vector([CD_H[0, 1], CD_H[1, 1], CD_H[2, 1]])
      par_CD_x3 = ti.Vector([CD_H[0, 2], CD_H[1, 2], CD_H[2, 2]])
      par_CD_x4 = -par_CD_x1 - par_CD_x2 - par_CD_x3
      sum_par_CD = w1 * par_CD_x1.norm_sqr() + w2 * par_CD_x2.norm_sqr(
      ) + w3 * par_CD_x3.norm_sqr() + w4 * par_CD_x4.norm_sqr()
      alpha_tilde_D = self.devia_alpha / (self.dt * self.dt * self.t_m[k])

      sum_par_CDH = w1 * par_CD_x1.dot(par_CH_x1) + w2 * par_CD_x2.dot(
          par_CH_x2) + w3 * par_CD_x3.dot(par_CH_x3) + w4 * par_CD_x4.dot(
              par_CH_x4)
      delta_lambda_H = (sum_par_CDH *
                        (C_D + alpha_tilde_D * self.devia_lambda[k]) -
                        (C_H + alpha_tilde_H * self.hydro_lambda[k]) *
                        (alpha_tilde_D + sum_par_CD)) / (
                            (alpha_tilde_H + sum_par_CH) *
                            (alpha_tilde_D + sum_par_CD) -
                            sum_par_CDH * sum_par_CDH)
      delta_lambda_D = -(C_D + alpha_tilde_D * self.devia_lambda[k] +
                         sum_par_CDH * delta_lambda_H) / (alpha_tilde_D +
                                                          sum_par_CD)

      self.v_p[a] += w1 * par_CH_x1 * delta_lambda_H
      self.v_p[b] += w2 * par_CH_x2 * delta_lambda_H
      self.v_p[c] += w3 * par_CH_x3 * delta_lambda_H
      self.v_p[d] += w4 * par_CH_x4 * delta_lambda_H
      self.hydro_lambda[k] += delta_lambda_H

      self.v_p[a] += w1 * par_CD_x1 * delta_lambda_D
      self.v_p[b] += w2 * par_CD_x2 * delta_lambda_D
      self.v_p[c] += w3 * par_CD_x3 * delta_lambda_D
      self.v_p[d] += w4 * par_CD_x4 * delta_lambda_D
      self.devia_lambda[k] += delta_lambda_D
