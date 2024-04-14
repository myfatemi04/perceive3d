import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
from set_axes_equal import set_axes_equal
from voxelize import voxelize


def find_valid_point(pcd, xy, r=10):
    roi_xyz = pcd[
        xy[1] - r : xy[1] + r,
        xy[0] - r : xy[0] + r
    ]
    mask = ~(roi_xyz == -10000).any(axis=-1)
    return np.median(roi_xyz[mask], axis=0)

def smoothen_pcd(pcd):
    pcd_smooth = cv2.GaussianBlur(pcd, (5, 5), 2)
    pcd_smooth[pcd == -10000] = -10000
    return pcd_smooth

with open("capture_2.pkl", "rb") as f:
    (rgbs, pcds) = pickle.load(f)

# smoothen pcds
pcds = [smoothen_pcd(pcd) for pcd in pcds]

plt.title("RGB Image")
plt.imshow(rgbs[0])
plt.axis('off')
plt.show()

target_point = tuple(int(x) for x in input("target: ").split())

# Select nearby points from the other point cloud.
center_xyz = find_valid_point(pcds[0], (750, 484))
# center_xyz = find_valid_point(pcds[0], (660, 516))

pcds[0][..., 0] += .05

# Select points within a certain radius of the median.
# Essentially, we segment the point cloud.
# To do this more accurately, at some point we want to use SAM.
radius = 0.3
all_points = np.concatenate([pcds[0].reshape(-1, 3), pcds[1].reshape(-1, 3)])
all_colors = np.concatenate([rgbs[0].reshape(-1, 3), rgbs[1].reshape(-1, 3)])
distances = np.linalg.norm(all_points - center_xyz, axis=-1)
mask = distances < radius
object_points = all_points[mask]
object_point_colors = all_colors[mask]

# Center the object.
object_points -= center_xyz

fig = plt.figure()
ax: plt.Axes = fig.add_subplot(projection='3d')
ax.scatter(object_points[:, 0], object_points[:, 1], object_points[:, 2], c=object_point_colors / 255, s=0.5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
set_axes_equal(ax)
plt.show()

lower_bound, upper_bound = np.min(object_points, axis=0), np.max(object_points, axis=0)

# Voxelize the point cloud.
voxelized = voxelize(object_points, object_point_colors, (lower_bound, upper_bound), 0.005)

voxel_occupancy = np.zeros(voxelized.shape[:-1], dtype=bool)
voxel_occupancy[voxelized[..., -1] > 0] = True
voxel_color = voxelized

fig = plt.figure()
ax: plt.Axes = fig.add_subplot(projection='3d')
ax.voxels(voxel_occupancy, facecolors=voxel_color, edgecolor=(1, 1, 1, 0.1))
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
set_axes_equal(ax)
plt.title("Voxelized Point Cloud")
plt.show()
