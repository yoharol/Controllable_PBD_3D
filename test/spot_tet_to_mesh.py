import meshio
import numpy as np

from utils import geom3d

mesh = meshio.read('assets/spot_high/spot_high.mesh')
weights = np.loadtxt('assets/spot_high/spot_high_w.txt')

verts = mesh.points
faces = mesh.cells_dict['triangle']
faces = faces[::2]
faces = faces.flatten()

n_surface_verts, vert_idx, face_idx, corres_idx = geom3d.extract_surface(
    verts, faces)
surface_weights = weights[vert_idx]
surface_verts = verts[vert_idx]

# write surface_verts and face_idx to obj file with meshio
# write surface_weights to txt file
meshio.write_points_cells(
    "assets/spot_mesh/spot_mesh.obj",
    surface_verts,
    [("triangle", face_idx.reshape(-1, 3))],
)
np.savetxt('assets/spot_mesh/spot_mesh_w.txt', surface_weights)