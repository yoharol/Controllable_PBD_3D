import taichi as ti
import numpy as np


# Linear Blend Skinning
@ti.data_oriented
class CompDynBase:

  def __init__(self,
               v_p: ti.MatrixField,
               v_p_ref: ti.MatrixField,
               v_p_rig: ti.MatrixField,
               v_invm: ti.Field,
               c_p: ti.MatrixField,
               c_p_ref: ti.MatrixField,
               v_weights: ti.Field,
               dt,
               alpha=0.0,
               alpha_fixed=0.0,
               fixed=[]) -> None:

    self.n_vert = v_p.shape[0]
    self.v_p = v_p
    self.v_invm = v_invm
    self.v_p_rig = v_p_rig
    self.v_p_ref = v_p_ref
    self.n_controls = c_p.shape[0]
    self.c_p = c_p
    self.c_p_ref = c_p_ref
    self.weights = v_weights
    self.lambdaf = ti.field(dtype=ti.f32, shape=(self.n_controls, 3))
    self.C = ti.field(dtype=ti.f32, shape=(self.n_controls, 3))
    self.delta_lambda = ti.field(dtype=ti.f32, shape=(self.n_controls, 3))
    self.alpha = alpha / (dt * dt)
    self.alpha_fixed = alpha_fixed / (dt * dt)
    self.alpha_list = ti.field(dtype=ti.f32, shape=(self.n_controls))
    self.sum_deriv = ti.field(dtype=ti.f32, shape=(self.n_controls, 3))
    self.sum_deriv_cache = ti.field(dtype=ti.f32, shape=(self.n_controls))

    self.set_fixed(fixed)

  def set_fixed(self, fixed):
    self.fixed = fixed
    self.alpha_list.fill(self.alpha)
    for i in fixed:
      self.alpha_list[i] = self.alpha_fixed

  def init_rest_status(self):
    self.compute_deriv_sum()

  def preupdate_cons(self):
    self.lambdaf.fill(0.0)

  def update_cons(self):
    self.solve_cons()

  @ti.kernel
  def compute_deriv_sum(self):
    self.sum_deriv_cache.fill(0.0)
    for i in range(self.n_vert):
      for j in range(self.n_controls):
        m = 1.0 / self.v_invm[i]
        w = self.weights[i, j]
        self.sum_deriv_cache[j] += m * w * w

  @ti.kernel
  def solve_cons(self):
    self.C.fill(0.0)
    self.sum_deriv.fill(0.0)

    ti.loop_config(serialize=True)
    for i in range(self.n_vert):
      x_c = self.v_p[i] - self.v_p_rig[i]
      for j in range(self.n_controls):
        m = 1.0 / self.v_invm[i]
        w = self.weights[i, j]
        self.C[j, 0] += m * w * x_c[0]
        self.C[j, 1] += m * w * x_c[1]
        self.C[j, 2] += m * w * x_c[2]

    ti.loop_config(serialize=True)
    for i in range(self.n_controls):
      self.sum_deriv[i, 0] = self.sum_deriv_cache[i]
      self.sum_deriv[i, 1] = self.sum_deriv_cache[i]
      self.sum_deriv[i, 2] = self.sum_deriv_cache[i]
      for j in range(3):
        self.delta_lambda[
            i,
            j] = -(self.C[i, j] + self.alpha_list[i] * self.lambdaf[i, j]) / (
                self.sum_deriv[i, j] + self.alpha_list[i])
        self.lambdaf[i, j] += self.delta_lambda[i, j]

    for i in range(self.n_vert):
      delta_x = ti.Vector([0.0, 0.0, 0.0])
      for j in range(self.n_controls):
        w = self.weights[i, j]
        delta_x += self.delta_lambda[j, 0] * w * ti.Vector([1.0, 0.0, 0.0])
        delta_x += self.delta_lambda[j, 1] * w * ti.Vector([0.0, 1.0, 0.0])
        delta_x += self.delta_lambda[j, 2] * w * ti.Vector([0.0, 0.0, 1.0])
      self.v_p[i] += delta_x


# Linear Blend Skinning
@ti.data_oriented
class CompDynMomentum:

  def __init__(self,
               v_p: ti.MatrixField,
               v_p_ref: ti.MatrixField,
               v_p_rig: ti.MatrixField,
               v_invm: ti.Field,
               c_p: ti.MatrixField,
               c_p_ref: ti.MatrixField,
               v_weights: ti.Field,
               dt,
               alpha=0.0,
               alpha_fixed=0.0,
               fixed=[]) -> None:

    self.n_vert = v_p.shape[0]
    self.n_controls = c_p.shape[0]
    self.n_fixed = len(fixed)
    self.n_free = self.n_controls - self.n_fixed
    self.v_p = v_p
    self.v_invm = v_invm
    self.v_p_rig = v_p_rig
    self.v_p_ref = v_p_ref
    self.c_p = c_p
    self.c_p_ref = c_p_ref
    self.weights = v_weights
    self.lambdaf = ti.field(dtype=ti.f32, shape=(self.n_controls, 3))
    self.C = ti.field(dtype=ti.f32, shape=(self.n_controls, 3))
    self.delta_lambda = ti.field(dtype=ti.f32, shape=(self.n_controls, 3))
    self.alpha = alpha / (dt * dt)
    self.alpha_fixed = alpha_fixed / (dt * dt)
    self.alpha_list = ti.field(dtype=ti.f32, shape=(self.n_controls))
    self.sum_deriv = ti.field(dtype=ti.f32, shape=(self.n_controls, 3))
    self.sum_deriv_cache = ti.field(dtype=ti.f32, shape=(self.n_controls))

    self.is_fixed = ti.field(dtype=ti.i32, shape=(self.n_controls))
    self.mass_center = ti.Vector.field(self.v_p.n, dtype=ti.f32, shape=())
    if self.n_fixed > 0:
      self.fix_idx = ti.field(dtype=ti.i32, shape=(self.n_fixed))
      self.fix_idx.from_numpy(np.array(fixed, dtype=np.int32))
      print("fixed idx", self.fix_idx)

    if self.n_free > 0:
      self.free_idx = ti.field(dtype=ti.i32,
                               shape=(self.n_controls - self.n_fixed))
      self.free_weight = ti.field(dtype=ti.f32, shape=(self.n_vert))
      self.momentum_deriv_sum = ti.field(dtype=ti.f32, shape=())

    self.set_fixed(fixed)

  def set_fixed(self, fixed):
    self.fixed = fixed
    self.alpha_list.fill(self.alpha)
    self.is_fixed.fill(0)
    for i in fixed:
      self.alpha_list[i] = self.alpha_fixed
      self.is_fixed[i] = 1
    if self.n_free > 0:
      free_list = []
      for i in range(self.n_controls):
        if self.is_fixed[i] == 0:
          free_list.append(i)
      self.free_idx.from_numpy(np.array(free_list, dtype=np.int32))

  def init_rest_status(self):
    self.compute_deriv_sum()

  def preupdate_cons(self):
    self.lambdaf.fill(0.0)

  def update_cons(self):
    self.C.fill(0.0)
    self.sum_deriv.fill(0.0)
    if self.n_fixed > 0:
      self.solve_cons_fix()
    if self.n_free > 0:
      self.compute_mass_center()
      self.solve_cons_free()
      self.fix_mass_center()

  @ti.kernel
  def compute_mass_center(self):
    mass_center = ti.Vector.zero(ti.f32, self.v_p.n)
    ti.loop_config(serialize=True)
    for i in range(self.n_vert):
      m = 1.0 / self.v_invm[i]
      w = self.free_weight[i]
      mass_center += m * w * self.v_p[i]
    self.mass_center[None] = mass_center

  @ti.kernel
  def fix_mass_center(self):
    new_mass_center = ti.Vector.zero(ti.f32, self.v_p.n)
    ti.loop_config(serialize=True)
    for i in range(self.n_vert):
      m = 1.0 / self.v_invm[i]
      w = self.free_weight[i]
      new_mass_center += m * w * self.v_p[i]
    delta = new_mass_center - self.mass_center[None]
    for i in range(self.n_vert):
      w = self.free_weight[i]
      self.v_p[i] += -w * delta / self.momentum_deriv_sum[None]

  @ti.kernel
  def compute_deriv_sum(self):
    self.sum_deriv_cache.fill(0.0)
    self.free_weight.fill(0.0)
    ti.loop_config(serialize=True)
    for i in range(self.n_vert):
      for j in range(self.n_controls):
        m = 1.0 / self.v_invm[i]
        w = self.weights[i, j]
        self.sum_deriv_cache[j] += m * w * w
      for jdx in range(self.n_free):
        j = self.free_idx[jdx]
        self.free_weight[i] += self.weights[i, j]
    self.momentum_deriv_sum[None] = 0.0
    ti.loop_config(serialize=True)
    for i in range(self.n_vert):
      m = 1.0 / self.v_invm[i]
      w = self.free_weight[i]
      self.momentum_deriv_sum[None] += m * w * w

  @ti.kernel
  def solve_cons_fix(self):
    ti.loop_config(serialize=True)
    for i in range(self.n_vert):
      x_c = self.v_p[i] - self.v_p_rig[i]
      for jdx in range(self.n_fixed):
        j = self.fix_idx[jdx]
        m = 1.0 / self.v_invm[i]
        w = self.weights[i, j]
        self.C[j, 0] += m * w * x_c[0]
        self.C[j, 1] += m * w * x_c[1]
        self.C[j, 2] += m * w * x_c[2]

    ti.loop_config(serialize=True)
    for idx in range(self.n_fixed):
      i = self.fix_idx[idx]
      self.sum_deriv[i, 0] = self.sum_deriv_cache[i]
      self.sum_deriv[i, 1] = self.sum_deriv_cache[i]
      self.sum_deriv[i, 2] = self.sum_deriv_cache[i]
      for j in range(3):
        self.delta_lambda[
            i,
            j] = -(self.C[i, j] + self.alpha_list[i] * self.lambdaf[i, j]) / (
                self.sum_deriv[i, j] + self.alpha_list[i])
        self.lambdaf[i, j] += self.delta_lambda[i, j]

    ti.loop_config(serialize=True)
    for i in range(self.n_vert):
      delta_x = ti.Vector([0.0, 0.0, 0.0])
      for jdx in range(self.n_fixed):
        j = self.fix_idx[jdx]
        w = self.weights[i, j]
        delta_x += self.delta_lambda[j, 0] * w * ti.Vector([1.0, 0.0, 0.0])
        delta_x += self.delta_lambda[j, 1] * w * ti.Vector([0.0, 1.0, 0.0])
        delta_x += self.delta_lambda[j, 2] * w * ti.Vector([0.0, 0.0, 1.0])
      self.v_p[i] += delta_x

  @ti.kernel
  def solve_cons_free(self):

    ti.loop_config(serialize=True)
    for i in range(self.n_vert):
      x_c = self.v_p[i] - self.v_p_rig[i]
      for jdx in range(self.n_free):
        j = self.free_idx[jdx]
        m = 1.0 / self.v_invm[i]
        w = self.weights[i, j]
        self.C[j, 0] += m * w * x_c[0]
        self.C[j, 1] += m * w * x_c[1]
        self.C[j, 2] += m * w * x_c[2]

    ti.loop_config(serialize=True)
    for idx in range(self.n_free):
      i = self.free_idx[idx]
      self.sum_deriv[i, 0] = self.sum_deriv_cache[i]
      self.sum_deriv[i, 1] = self.sum_deriv_cache[i]
      self.sum_deriv[i, 2] = self.sum_deriv_cache[i]
      for j in range(3):
        self.delta_lambda[
            i,
            j] = -(self.C[i, j] + self.alpha_list[i] * self.lambdaf[i, j]) / (
                self.sum_deriv[i, j] + self.alpha_list[i])
        self.lambdaf[i, j] += self.delta_lambda[i, j]

    ti.loop_config(serialize=True)
    for i in range(self.n_vert):
      delta_x = ti.Vector([0.0, 0.0, 0.0])
      for jdx in range(self.n_free):
        j = self.free_idx[jdx]
        w = self.weights[i, j]
        delta_x += self.delta_lambda[j, 0] * w * ti.Vector([1.0, 0.0, 0.0])
        delta_x += self.delta_lambda[j, 1] * w * ti.Vector([0.0, 1.0, 0.0])
        delta_x += self.delta_lambda[j, 2] * w * ti.Vector([0.0, 0.0, 1.0])
      self.v_p[i] += delta_x