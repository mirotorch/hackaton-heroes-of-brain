import mne
import numpy as np
from julia import JuliaSetAnimation

raw = mne.io.read_raw_fif('left_ernest.fif', preload=True)
data = raw.get_data()[:4, :]
data = data.T  

def normalize(i, constant):
    return (constant, np.sin(i * 0.25 + 0.6))

def prepare_data(data, real_constants):
    new_data = []
    for row in data:
        new_row = []
        for i in range(4):
            new_row.append(normalize(row[i], real_constants[i]))
        new_data.append(new_row)
    return new_data

real_constants = [-0.2, -0.5, -0.7, -1.0]

julia_anim = JuliaSetAnimation()

new_data = prepare_data(data, real_constants)

julia_anim.add_data(new_data)

julia_anim.animate(frames=len(new_data), interval=1)
