import numpy as np
import meshio

def load_obj(filename: str):
  mesh = meshio.read(filename)
  verts = mesh.points
  assert 'triangle' in mesh.cells_dict, 'only triangle mesh is supported'
  faces = mesh.cells_dict['triangle']
  return verts, faces.flatten()

def load_tet(filename: str):
  mesh = meshio.read(filename)
  verts = mesh.points
  assert 'tetra' in mesh.cells_dict, 'only tetra mesh is supported'
  tets = mesh.cells_dict['tetra']
  faces = mesh.cells_dict['triangle']
  return verts, tets.flatten(), faces.flatten()

def image_loader(filepath):
  from PIL import Image

  image_frame = np.array(Image.open(filepath))
  image_frame = np.flip(image_frame.swapaxes(0, 1), 1)
  image_frame = image_frame[..., 0:3]
  return image_frame

def tgf_loader(filename: str):
  points = []
  cage_edges = []
  bone_edges = []

  with open(filename, 'r') as f:
    lines = f.readlines()
    state = 0
    for line in lines:
      line = line.replace('\n', '')
      if line[0] == '#':
        state += 1
      elif state == 0:
        point_data = str.split(line, sep=' ')
        p = [float(point_data[1]), float(point_data[2]), float(point_data[3])]
        points.append(p)
      elif state == 1:
        edge_data = str.split(line, sep=' ')
        e = [int(edge_data[0]), int(edge_data[1])]
        e_type = int(edge_data[2])
        if e_type == 1:
          bone_edges.append(e)
        elif e_type == 2:
          cage_edges.append(e)
  return np.array(points, dtype=np.float32), np.array(bone_edges, dtype=np.int32).flatten(), np.array(cage_edges, dtype=np.int32).flatten()
