import numpy as np
import taichi as ti


@ti.data_oriented
class MotionData3D:

  def __init__(self,
               datapath: str,
               c_A_input: ti.MatrixField,
               c_b_input: ti.MatrixField,
               idx: list = None,
               scale=1.0) -> None:
    vec_data = []
    rotvec_data = []
    num_bones = -1

    with open(datapath, 'r') as f:
      lines = f.readlines()
      frames = 0
      for line in lines:
        if len(line) > 10:
          rawdata = line.split(' ')
          # assert frames == int(rawdata[0]), f'{frames} {rawdata[0]}'
          frames += 1
          base = 1
          tmp = (len(rawdata) - base) // 7
          if num_bones == -1:
            num_bones = tmp
          else:
            assert num_bones == tmp
          vec = []
          rot = []
          for i in range(num_bones):
            vec.append(float(rawdata[base + i * 7]))
            vec.append(float(rawdata[base + i * 7 + 1]))
            vec.append(float(rawdata[base + i * 7 + 2]))
            r = float(rawdata[base + i * 7 + 3])
            rot.append(float(rawdata[base + i * 7 + 4]) * r)
            rot.append(float(rawdata[base + i * 7 + 5]) * r)
            rot.append(float(rawdata[base + i * 7 + 6]) * r)
          vec_data.append(vec)
          rotvec_data.append(rot)

    self.vec_data = np.array(vec_data, dtype=np.float32)
    assert self.vec_data.shape == (frames, num_bones * 3)
    self.vec_data = self.vec_data.reshape((frames, num_bones, 3)) * scale
    self.rotvec_data = np.array(rotvec_data, dtype=np.float32)
    assert self.rotvec_data.shape == (frames, num_bones * 3)
    self.rotvec_data = self.rotvec_data.reshape((frames, num_bones, 3))

    self.frames = frames
    self.n_bones = num_bones

    self.vec = ti.Vector.field(3, dtype=ti.f32, shape=self.n_bones)
    self.rotvec = ti.Vector.field(3, dtype=ti.f32, shape=self.n_bones)

    if idx == None:
      self.idx = list(range(self.n_bones))

    self.read_data(0)

  def read_data(self, frame):
    frame = frame % self.frames
    self.vec.from_numpy(self.vec_data[frame])
    self.rotvec.from_numpy(self.rotvec_data[frame])
