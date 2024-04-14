import pickle

import matplotlib.pyplot as plt
import numpy as np
from set_axes_equal import set_axes_equal

def find_valid_point(pcd, xy, r=10):
    roi_xyz = pcd[
        xy[1] - r : xy[1] + r,
        xy[0] - r : xy[0] + r
    ]
    mask = ~(roi_xyz == -10000).any(axis=-1)
    return np.median(roi_xyz[mask], axis=0)

with open("capture_0.pkl", "rb") as f:
    (rgbs, pcds) = pickle.load(f)

plt.title("RGB Image")
plt.imshow(rgbs[0])
plt.axis('off')
plt.show()

# generate a small radius around the target object
# target_location = (771, 455)

# width = 50
# height = 120
# roi_xyz = pcds[0][
#     target_location[1] - height // 2 : target_location[1] + height // 2,
#     target_location[0] - width // 2 : target_location[0] + width // 2
# ]
# roi_rgb = rgbs[0][
#     target_location[1] - height // 2 : target_location[1] + height // 2,
#     target_location[0] - width // 2 : target_location[0] + width // 2
# ]
# mask = ~(roi_xyz == -10000).any(axis=-1)
# roi_xyz = roi_xyz[mask]
# roi_rgb = roi_rgb[mask]

# Select nearby points from the other point cloud.
# center_xyz = np.median(roi_xyz, axis=0, keepdims=True)
center_xyz = find_valid_point(pcds[0], (660, 516))

pcds[0][..., 0] += .05

# Select points within a certain radius of the median.
radius = 0.1
all_points = np.concatenate([pcds[0].reshape(-1, 3), pcds[1].reshape(-1, 3)])
all_colors = np.concatenate([rgbs[0].reshape(-1, 3), rgbs[1].reshape(-1, 3)])
distances = np.linalg.norm(all_points - center_xyz, axis=-1)
mask = distances < radius
roi_xyz = all_points[mask]
roi_rgb = all_colors[mask]

fig = plt.figure()
ax: plt.Axes = fig.add_subplot(projection='3d')
ax.scatter(roi_xyz[:, 0], roi_xyz[:, 1], roi_xyz[:, 2], c=roi_rgb / 255, s=0.5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
set_axes_equal(ax)
plt.show()
