from OpenGL.GL import *
from PIL import Image
import numpy as np
import glfw
import time
import glfw.GLFW
import taichi as ti
from utils.gl import *
from utils.geom2d import edge_extractor


class DragEvent:

  def __init__(self) -> None:
    self.dragging = False
    self.start_drag_pos = np.zeros(shape=(2,), dtype=np.float32)


class MeshRender3D:

  def __init__(self, title: str, res: tuple, kernel: str) -> None:
    if kernel == 'glfw':
      # self.kernel = 'glfw'
      # self.renderer = GlfwRender2D(title, res)
      assert False, 'MeshRender3D: glfw kernel is not supported'
    elif kernel == 'taichi':
      self.kernel = 'taichi'
      self.renderer = TaichiRender3D(title, res)
    else:
      print(f"error! kernel {kernel} is not supported")
      exit()
    self.title = title
    self.res = res
    self.drag_left = self.renderer.drag_left
    self.drag_right = self.renderer.drag_right
    self.time_rcd = self.renderer.time_rcd
    self.frame_rcd = self.renderer.frame_rcd
    self.wireframe_mode = False
    self.wireframe_color = (0.1, 0.4, 1.0, 1.0)

    self.key_events = []

    self.add_key_event('f', 'press', self.set_wireframe_mode)

  def set_camera(self,
                 eye: tuple = (3, 3, 3),
                 center: tuple = (0, 0, 0),
                 up: tuple = (0, 1, 0)):
    self.renderer.set_camera(eye, center, up)

  def get_cursor_pos(self):
    return self.renderer.get_cursor_pos()

  def running(self):
    return self.renderer.running

  def get_time(self):
    return self.time_rcd[1]

  def get_total_frames(self):
    return self.frame_rcd[1]

  def add_key_event(self, key_char: str, action: str, callback_func):
    assert len(key_char) == 1
    self.renderer.add_key_event(key_char, action.lower(), callback_func)

  def add_cursor_event(self, button_str: str, action: str, callback_func):
    self.renderer.add_cursor_event(button_str, action.lower(), callback_func)

  def set_wireframe_mode(self):
    self.wireframe_mode = not self.wireframe_mode
    self.renderer.set_wireframe_mode(self.wireframe_mode, self.wireframe_color)

  def set_background_color(self, color=(1.0, 1.0, 1.0, 1.0)):
    self.renderer.set_background_color(color)

  def add_render_func(self, render_func):
    self.renderer.add_render_func(render_func)

  def update_mesh(self, verts):
    self.renderer.update_mesh(verts)

  def render(self):
    self.renderer.render()

  def pre_update(self):
    self.renderer.pre_update()

  def show(self):
    self.renderer.show()

  def terminate(self):
    self.renderer.terminate()

  def get_screenshot(self, path: str):
    self.renderer.get_screen_shot(path)


class TaichiRender3D:

  def __init__(self, title: str, res: tuple) -> None:
    self.title = title
    self.res = res
    self.window = ti.ui.Window(name=self.title, res=self.res, fps_limit=60)
    self.canvas = self.window.get_canvas()
    self.gui = self.window.get_gui()
    self.scene = ti.ui.Scene()
    self.cursor_pos = np.zeros(shape=(2,), dtype=np.float32)
    self.cursor_events_press = []
    self.cursor_events_release = []
    self.key_events_press = []
    self.key_events_release = []
    self.running = True
    self.wireframe = False

    self.drag_left = DragEvent()
    self.drag_right = DragEvent()
    self.begin_time = time.time()
    self.time_rcd = [0.0, 0.0]  # [last_time, current_time]
    self.frame_rcd = [0, 0]  # [fps, total frames count]

    self.background_color = (1.0, 1.0, 1.0)

    self.camera = ti.ui.Camera()
    self.camera.position(3, 3, 3)
    self.camera.lookat(0, 0, 0)
    self.camera.up(0, 1, 0)

    self.render_funcs = []

  def pre_update(self):
    self.running = self.window.running
    self.cursor_pos = np.array(self.window.get_cursor_pos(), dtype=np.float32)
    self.camera.track_user_inputs(self.window,
                                  movement_speed=0.03,
                                  hold_key=ti.ui.RMB)
    events = self.window.get_events(ti.ui.PRESS)
    for e in events:
      if e.key == ti.ui.ESCAPE:
        self.running = False
        exit()
      if e.key == ti.ui.LMB:
        self.drag_left.start_drag_pos = self.cursor_pos
        self.drag_left.dragging = True
      elif e.key == ti.ui.RMB:
        self.drag_right.start_drag_pos = self.cursor_pos
        self.drag_right.dragging = True
      for event in self.cursor_events_press:
        if e.key == event[0]:
          event[1]()
      for event in self.key_events_press:
        if e.key == event[0]:
          event[1]()
    events = self.window.get_events(ti.ui.RELEASE)
    for e in events:
      if e.key == ti.ui.LMB:
        self.drag_left.dragging = False
      elif e.key == ti.ui.RMB:
        self.drag_right.dragging = False
      for event in self.cursor_events_release:
        if e.key == event[0]:
          event[1]()
      for event in self.key_events_release:
        if e.key == event[0]:
          event[1]()

  def add_cursor_event(self, button_str: str, action_str: str, callback_func):
    if button_str == 'left':
      button = ti.ui.LMB
    elif button_str == 'right':
      button = ti.ui.RMB
    if action_str == 'press':
      self.cursor_events_press.append((button, callback_func))
    elif action_str == 'release':
      self.cursor_events_release.append((button, callback_func))

  def add_key_event(self, key_char: str, action_str: str, callback_func):
    if action_str == 'press':
      self.key_events_press.append((key_char.lower(), callback_func))
    elif action_str == 'release':
      self.key_events_release.append((key_char.lower(), callback_func))

  def get_cursor_pos(self):
    return self.cursor_pos

  def set_camera(self, eye, center, up):
    self.camera.position(eye)
    self.camera.lookat(center)
    self.camera.up(up)

  def add_render_func(self, render_func):
    self.render_funcs.append(render_func)

  def set_background_color(self, color: tuple = (1.0, 1.0, 1.0, 1.0)):
    self.background_color = (color[0], color[1], color[2])

  def render(self):

    self.scene.set_camera(self.camera)
    self.scene.ambient_light((0.2, 0.2, 0.2))
    self.scene.point_light(pos=(4.0, 4.0, 4.0), color=(0.96, 0.96, 0.96))

    for func in self.render_funcs:
      func(self.scene)

  def show(self):
    self.canvas.scene(self.scene)
    self.window.show()
    self.time_rcd[0] = self.time_rcd[1]
    self.time_rcd[1] = time.time() - self.begin_time
    self.frame_rcd[0] = int(1.0 / (self.time_rcd[1] - self.time_rcd[0]))
    self.frame_rcd[1] += 1

  def get_screen_shot(self, path: str):
    self.window.save_image(path)

  def terminate(self):
    pass


class GlfwRender3D:

  def __init__(self) -> None:
    pass

  def pre_update(self):
    pass

  def add_cursor_event(self, button_str: str, action_str: str, callback_func):
    pass

  def add_key_event(self, key_char: str, action_str: str, callback_func):
    pass

  def get_cursor_pos(self):
    pass

  def set_camera(self, eye, center, up):
    pass

  def add_render_func(self, render_func):
    pass

  def set_background_color(self, color: tuple = (1.0, 1.0, 1.0, 1.0)):
    pass

  def render(self):
    pass

  def show(self):
    pass

  def get_screen_shot(self, path: str):
    pass

  def terminate(self):
    pass
