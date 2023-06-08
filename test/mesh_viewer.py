import pyvista

modelname = 'cube_tet.mesh'
plotmesh = pyvista.read(f"assets/{modelname}")
plotmesh.plot(smooth_shading=True, show_edges=True)