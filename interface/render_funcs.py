import taichi as ti
from data import cage_data, tet_data, cloth_data, points_data


def get_cage_render_func(cage: cage_data.CageData,
                         point_color=(0.0, 1.0, 0.0),
                         fixed_color=(1.0, 0.0, 0.0),
                         point_radius=0.1,
                         edge_color=(0.0, 1.0, 1.0),
                         edge_width=0.01,
                         fixed_points=[],
                         render_points=True,
                         render_edges=True):

  def render_func(scene: ti.ui.Scene):
    if render_edges:
      scene.lines(cage.c_p,
                  indices=cage.e_i,
                  width=edge_width,
                  color=edge_color)
    if render_points:
      scene.particles(cage.c_p,
                      radius=point_radius,
                      per_vertex_color=cage.c_color)

  return render_func


def get_points_render_func(points: points_data.PointsData,
                           point_color=(0.0, 1.0, 0.0),
                           fixed_color=(1.0, 0.0, 0.0),
                           point_radius=0.1,
                           fixed_points=[]):

  def render_func(scene: ti.ui.Scene):
    scene.particles(points.c_p,
                    radius=point_radius,
                    per_vertex_color=points.c_color)

  return render_func


def get_particles_render_func(points: ti.MatrixField,
                              color=(1.0, 0.0, 0.0),
                              radius=0.1,
                              per_vertex_color=None):
  if per_vertex_color is None:

    def render_func(scene: ti.ui.Scene):
      scene.particles(points, radius=radius, color=color)

    return render_func
  else:

    def render_func(scene: ti.ui.Scene):
      scene.particles(points, radius=radius, per_vertex_color=per_vertex_color)

    return render_func


def get_mesh_render_func(v_p: ti.MatrixField,
                         f_i: ti.Field,
                         wireframe: list,
                         color=(1.0, 1.0, 1.0)):

  def render_func(scene: ti.ui.Scene, wire=wireframe):
    scene.mesh(v_p, f_i, color=color, show_wireframe=wire[0])

  return render_func
