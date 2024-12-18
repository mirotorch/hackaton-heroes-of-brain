import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import queue

class JuliaSetAnimation:
    def __init__(self, xmin=-2, xmax=2, ymin=-2, ymax=2, width=800, height=800, max_iter=256, buffer_size=5000):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.width = width
        self.height = height
        self.max_iter = max_iter
        
        self.fig, self.axs = plt.subplots(2, 2, figsize=(10, 10))
        self.axs = self.axs.flatten()
        for ax in self.axs:
            ax.set_axis_off()
        self.img_displays = [ax.imshow(np.zeros((self.height, self.width)), extent=(self.xmin, self.xmax, self.ymin, self.ymax), cmap="viridis") for ax in self.axs]
        
        self.data_buffer = queue.Queue(maxsize=buffer_size)
        
    def julia_set(self, c):
        x = np.linspace(self.xmin, self.xmax, self.width)
        y = np.linspace(self.ymin, self.ymax, self.height)
        X, Y = np.meshgrid(x, y)
        Z = X + 1j * Y
        img = np.zeros(Z.shape, dtype=int)
        
        for i in range(self.max_iter):
            mask = np.abs(Z) < 2
            img[mask] = i
            Z[mask] = Z[mask]**2 + c
        return img

    def update(self, frame):
        if self.data_buffer.empty():
            plt.close(self.fig)
            return []

        for i, (real, imag) in enumerate(self.data_buffer.get()):
            c = complex(real, imag)
            img = self.julia_set(c)
            self.img_displays[i].set_data(img)
            self.img_displays[i].set_clim(0, self.max_iter) 
        return self.img_displays

    def animate(self, frames=100, interval=100):
        anim = FuncAnimation(self.fig, self.update, frames=frames, interval=interval, blit=True)
        plt.show()

    def add_data(self, new_data):
        for data in new_data:
            if self.data_buffer.full():
                self.data_buffer.get()
            self.data_buffer.put(data)
