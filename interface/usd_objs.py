from pxr import Usd, UsdGeom
from data import cage_data, tet_data, cloth_data
import numpy as np


class SaveMesh:

  def __init__(self,
               stage: Usd.Stage,
               prim_path: str,
               verts_np: np.ndarray,
               faces_np: np.ndarray,
               two_sided=False,
               color=(1.0, 1.0, 1.0)) -> None:
    assert faces_np.ndim == 1, 'faces should be 1D array'
    self.stage = stage
    self.geom = UsdGeom.Mesh.Define(stage, prim_path)
    self.geom.GetPointsAttr().Set(verts_np)
    self.geom.GetFaceVertexCountsAttr().Set([3] * (faces_np.shape[0] // 3))
    self.geom.GetFaceVertexIndicesAttr().Set(faces_np.reshape((-1, 3)))
    self.geom.GetDoubleSidedAttr().Set(two_sided)
    self.geom.GetDisplayColorAttr().Set([color])

  def update(self, verts_np: np.ndarray, timecode=0):
    self.geom.GetPointsAttr().Set(verts_np, time=timecode)


class SaveLines:

  def __init__(self,
               stage: Usd.Stage,
               prim_path: str,
               verts_np: np.ndarray,
               edges_np: np.ndarray,
               width=0.1,
               color=(1.0, 1.0, 1.0)) -> None:
    self.lines = []
    self.edges = edges_np.reshape((-1, 2))
    print(self.edges)
    self.n_edges = self.edges.shape[0]
    self.width = width
    self.xform = UsdGeom.Xform.Define(stage, prim_path)

    for i in range(self.n_edges):
      line = UsdGeom.BasisCurves.Define(stage, prim_path + "/line" + str(i))
      line.GetTypeAttr().Set(UsdGeom.Tokens.linear)
      line.GetCurveVertexCountsAttr().Set([2])
      line.GetWidthsAttr().Set([width])
      line.SetWidthsInterpolation(UsdGeom.Tokens.constant)
      line.GetPointsAttr().Set(verts_np[self.edges[i]])
      line.GetDisplayColorAttr().Set([color])
      self.lines.append(line)

  def update(self, verts_np: np.ndarray, timecode=0):
    for i in range(self.n_edges):
      self.lines[i].GetPointsAttr().Set(verts_np[self.edges[i]], time=timecode)


class SavePoints:

  def __init__(self,
               stage: Usd.Stage,
               prim_path: str,
               verts_np: np.ndarray,
               per_vert_color: np.ndarray = None,
               color=(1.0, 1.0, 1.0),
               radius=0.3) -> None:
    self.points = []
    self.n_points = verts_np.shape[0]

    self.xform = UsdGeom.Xform.Define(stage, prim_path)
    for i in range(self.n_points):
      point = UsdGeom.Sphere.Define(stage, prim_path + "/point" + str(i))
      UsdGeom.XformCommonAPI(point).SetTranslate(verts_np[i].tolist())
      point.GetRadiusAttr().Set(radius)
      if per_vert_color is None:
        point.GetDisplayColorAttr().Set([color])
      else:
        assert per_vert_color.shape == (self.n_points, 3)
        point.GetDisplayColorAttr().Set([per_vert_color[i].tolist()])
      self.points.append(point)

  def update(self, verts_np: np.ndarray, timecode=0):
    for i in range(self.n_points):
      UsdGeom.XformCommonAPI(self.points[i]).SetTranslate(verts_np[i].tolist(),
                                                          time=timecode)
