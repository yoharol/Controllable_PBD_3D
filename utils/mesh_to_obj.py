import meshio

mesh = meshio.read('assets/cube_tet/cube_tet.mesh')

# write verts and faces in mesh to obj by meshio
meshio.write_points_cells("assets/cube_tet/cube_tet.obj", mesh.points, [("triangle", mesh.cells_dict["triangle"])])
print(mesh.cells_dict["triangle"])

