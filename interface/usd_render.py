from pxr import Usd, UsdGeom


class UsdRender:

  def __init__(self, filepath: str, startTimeCode: int, endTimeCode: int,
               fps: int, UpAxis: str) -> None:
    self.stage = Usd.Stage.CreateNew(filepath)
    self.stage.SetStartTimeCode(startTimeCode)
    self.stage.SetEndTimeCode(endTimeCode)
    self.stage.SetTimeCodesPerSecond(fps)

    self.startTimeCode = startTimeCode
    self.endTimeCode = endTimeCode

    if UpAxis == "Y":
      UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.y)
    elif UpAxis == "Z":
      UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.z)

    self.rootXform = UsdGeom.Xform.Define(self.stage, "/root")
    self.rootPrim = self.stage.GetPrimAtPath("/root")
    UsdGeom.XformCommonAPI(self.rootPrim).SetScale((100.0, 100.0, 100.0))
    self.stage.SetDefaultPrim(self.rootPrim)

  def save(self):
    self.stage.Save()