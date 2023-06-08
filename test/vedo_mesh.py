from vedo import *

mesh = Mesh('out/cube_balloon_drag.obj')
mesh.color('aqua').c('aqua').lighting('glossy')

# Show the mesh, the two planes, the docstring
show(mesh, __doc__, viewup='y').close()
