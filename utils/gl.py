from OpenGL.GL import *
from OpenGL.GL import shaders
from PIL import Image
import numpy as np


class GL_VAO:

  def __init__(self) -> None:
    self.vao = glGenVertexArrays(1)

  def bind(self):
    glBindVertexArray(self.vao)

  def unbind(self):
    glBindVertexArray(0)


class GL_Buffer:

  def __init__(self, data: np.ndarray, target, draw_type) -> None:
    self.target = target
    self.buffer = glGenBuffers(1)
    self.draw_type = draw_type

    self.update_buffer_data(data)

  def bind(self):
    glBindBuffer(self.target, self.buffer)

  def unbind(self):
    glBindBuffer(self.target, 0)

  def update_buffer_data(self, data: np.ndarray):
    self.bind()
    glBufferData(self.target, data.nbytes, data, self.draw_type)
    self.unbind()


class GL_Texture2D:

  def __init__(self, filepath: str, img_type) -> None:
    self.texture = glGenTextures(1)
    self.bind()

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
    #glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    #glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    image = Image.open(filepath)
    image = image.convert('RGBA')
    img_data = np.array(list(image.getdata()), np.uint8)
    image_size = image.size

    image_width = image_size[0]
    image_height = image_size[1]

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    glTexImage2D(GL_TEXTURE_2D, 0, img_type, image_width, image_height, 0,
                 img_type, GL_UNSIGNED_BYTE, img_data)
    glGenerateMipmap(GL_TEXTURE_2D)

    self.unbind()

  def bind(self):
    glBindTexture(GL_TEXTURE_2D, self.texture)

  def unbind(self):
    glBindTexture(GL_TEXTURE_2D, 0)


class Mesh2dShaderProgram:

  def __init__(self, tex_channel=0) -> None:
    self.tex_channel = tex_channel
    self.program = self.mesh2d_shader_program()
    self.bind()
    self.pos_attri = glGetAttribLocation(self.program, "inPos")
    self.uv_attri = glGetAttribLocation(self.program, "inUVs")
    glUniform1i(glGetUniformLocation(self.program, "imageTexture"),
                self.tex_channel)
    glUniform1i(glGetUniformLocation(self.program, "wireframe"), False)
    self.wireframe = False
    self.unbind()

  def set_wireframe_mode(self,
                         wireframe: bool,
                         color: tuple = (0.0, 0.0, 0.0, 1.0)):
    self.bind()
    self.wireframe = wireframe
    if wireframe:
      glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
    else:
      glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

    glUniform1i(glGetUniformLocation(self.program, "wireframe"), wireframe)
    glUniform4f(glGetUniformLocation(self.program, "wireframe_color"), color[0],
                color[1], color[2], color[3])
    self.unbind()

  def bind(self):
    glUseProgram(self.program)

  def unbind(self):
    glUseProgram(0)

  def bind_buffer(self, vert_buffer: GL_Buffer, uv_buffer: GL_Buffer):
    vert_buffer.bind()
    glEnableVertexAttribArray(self.pos_attri)
    glVertexAttribPointer(self.pos_attri, 2, GL_FLOAT, GL_FALSE, 0,
                          ctypes.c_void_p(0))
    vert_buffer.unbind()
    uv_buffer.bind()
    glEnableVertexAttribArray(self.uv_attri)
    glVertexAttribPointer(self.uv_attri, 2, GL_FLOAT, GL_FALSE, 0,
                          ctypes.c_void_p(0))
    uv_buffer.unbind()

  def bind_texture(self, texture: GL_Texture2D):
    glActiveTexture(GL_TEXTURE0)
    texture.bind()

  def mesh2d_shader_program(self):
    vertex_shader = """#version 330 core
        in vec2 inPos;
        in vec2 inUVs;
        // in vec4 colors;
        out vec2 outUVs;

        void main()
        {
            gl_Position = vec4(inPos*2.0-1.0, 0.0, 1.0);
            outUVs = vec2(inUVs[0], 1.0 - inUVs[1]);
        }
      """
    fragment_shader = """#version 330 core
        in vec2 outUVs;
        out vec4 color;

        uniform sampler2D imageTexture;
        uniform bool wireframe;
        uniform vec4 wireframe_color;

        void main()
        {
            if (wireframe){
              color = wireframe_color;
            }
            else{
              color = texture(imageTexture, outUVs);
            }
        }
      """
    return shaders.compileProgram(
        shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
        shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER))


class GL_Mesh2D:

  def __init__(self, verts, uvs, faces, dynamic_verts=True) -> None:
    self.vert_draw_type = GL_DYNAMIC_DRAW if dynamic_verts else GL_STATIC_DRAW
    self.vao = GL_VAO()
    self.vao.bind()
    self.n_faces = faces.size // 3
    self.vert_buffer = GL_Buffer(verts.flatten(),
                                 target=GL_ARRAY_BUFFER,
                                 draw_type=self.vert_draw_type)
    self.uv_buffer = GL_Buffer(uvs.flatten(),
                               target=GL_ARRAY_BUFFER,
                               draw_type=GL_STATIC_DRAW)
    self.face_buffer = GL_Buffer(faces.flatten(),
                                 target=GL_ELEMENT_ARRAY_BUFFER,
                                 draw_type=GL_STATIC_DRAW)

  def update_verts(self, verts: np.ndarray):
    assert self.vert_draw_type == GL_DYNAMIC_DRAW
    self.vert_buffer.update_buffer_data(verts.flatten())

  def bind(self):
    self.vao.bind()
    self.face_buffer.bind()

  def unbind(self):
    self.face_buffer.unbind()
    self.vao.unbind()

  def draw(self):
    glDrawElements(GL_TRIANGLES, self.n_faces * 3, GL_UNSIGNED_INT, None)
