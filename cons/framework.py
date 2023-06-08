import taichi as ti
from utils.mathlib import isnan


@ti.data_oriented
class pbd_framework:

  def __init__(self,
               v_p: ti.MatrixField,
               g: ti.Vector,
               dt: float,
               damp: float = 1.0) -> None:
    self.n_verts = v_p.shape[0]
    self.v_p = v_p
    self.v_p_cache = ti.Vector.field(v_p.n, dtype=v_p.dtype, shape=self.n_verts)
    self.v_p_cache = ti.Vector.field(v_p.n, dtype=v_p.dtype, shape=self.n_verts)
    self.v_v = ti.Vector.field(v_p.n, dtype=v_p.dtype, shape=self.n_verts)
    self.dt = dt
    self.g = g
    self.damp = ti.field(dtype=ti.f32, shape=())
    self.damp[None] = damp

    self.cons_list = [[]]
    self.initupdate = []
    self.preupdate = []
    self.collision = []

  @ti.kernel
  def make_prediction(self):
    ti.loop_config(serialize=True)
    for k in range(self.n_verts):
      self.v_p_cache[k] = self.v_p[k]
      self.v_p[
          k] = self.v_p[k] + self.v_v[k] * self.dt + self.g * self.dt * self.dt

  @ti.kernel
  def update_vel(self):
    ti.loop_config(serialize=True)
    for k in range(self.n_verts):
      self.v_v[k] = self.damp[None] * (self.v_p[k] -
                                       self.v_p_cache[k]) / self.dt

  def add_cons(self, new_cons, index=0):
    if index > len(self.cons_list):
      print('wrong constraints index')
      return
    if index == len(self.cons_list):
      self.cons_list.append([])
    self.cons_list[index].append(new_cons)

  def add_preupdate(self, preupdate):
    self.preupdate.append(preupdate)

  def add_init(self, init):
    self.initupdate.append(init)

  def add_collision(self, obj):
    self.collision.append(obj)

  def init_rest_status(self, index=0):
    for init in self.initupdate:
      init()
    for i in range(len(self.cons_list[index])):
      self.cons_list[index][i].init_rest_status()
    for coll in self.collision:
      coll(self.v_p)

  def preupdate_cons(self, index=0):
    for update in self.preupdate:
      update()
    if index >= len(self.cons_list):
      return
    for i in range(len(self.cons_list[index])):
      self.cons_list[index][i].preupdate_cons()
    for coll in self.collision:
      coll(self.v_p)

  def update_cons(self, index=0):
    if index >= len(self.cons_list):
      return
    for i in range(len(self.cons_list[index])):
      self.cons_list[index][i].update_cons()
