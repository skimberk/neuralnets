from tkinter import *
from PIL import Image, ImageDraw
import math
import numpy as np

class _DigitDrawerGUI(Frame):
    def __init__(self, dimensions, listener):
        Frame.__init__(self)

        self.dimensions = dimensions
        self.listener = listener

        self.pack(expand=YES, fill=BOTH)
        self.master.title('Digit Drawer')
        self.master.geometry('x'.join(map(str, dimensions)))
        self.master.resizable(0, 0)

        # Create Canvas component
        self.canvas = Canvas(self)
        self.canvas.pack(expand = YES, fill = BOTH)

        # Bind mouse dragging event to Canvas
        self.canvas.bind('<B1-Motion>', self.move)
        self.canvas.bind('<ButtonRelease-1>', self.release)

        self.line_counter = 0
        self.line_ids = []
        self.line_coords = []

        self.drawing = False
        self.last_x = 0
        self.last_y = 0

    def draw_temporary_line(self, x1, y1, x2, y2):
        line_id = 'line_' + str(self.line_counter)
        self.line_ids.append(line_id)
        self.line_counter += 1

        self.canvas.create_line(x1, y1, x2, y2, width=20, tag=line_id)

    def make_temporary_line_permanent(self):
        if self.line_counter > 1:
            for line_id in self.line_ids:
                self.canvas.delete(line_id)

            self.canvas.create_line(*self.line_coords, width=20)
            self.listener._draw_line(self.line_coords)

        # Empty the lists
        self.line_ids[:] = []
        self.line_coords[:] = []

        self.line_counter = 0

    def move(self, event):
        if self.drawing:
            self.draw_temporary_line(self.last_x, self.last_y, event.x, event.y)
        else:
            self.drawing = True

        self.last_x = event.x
        self.last_y = event.y

        self.line_coords.append((event.x, event.y))

    def release(self, event):
        self.make_temporary_line_permanent()
        self.drawing = False

class DigitDrawer():
    def __init__(self, on_update):
        self.final_size = (8, 8)
        self.scale_factor = 40
        self.size = (self.final_size[0] * self.scale_factor, self.final_size[1] * self.scale_factor)
        self.on_update = on_update

        self.image = Image.new('L', self.size, 0)
        self.draw = ImageDraw.Draw(self.image)

        self.gui = _DigitDrawerGUI(self.size, self)
        self.gui.mainloop()

    def _draw_line(self, line_coords):
        self.draw.line(line_coords, fill=255, width=20)

        thumbnail = self.image.copy()
        thumbnail.thumbnail(self.final_size)
        self.on_update(np.array(thumbnail) / 255)
