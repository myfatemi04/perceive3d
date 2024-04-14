import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d
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

def get_normal(window):
    # we use finite differences method.
    # window is a 3x3 matrix.
    dx = (window[2, 1] - window[0, 1]) / 2
    dy = (window[1, 2] - window[1, 0]) / 2
    normal = np.array([dx, dy, 1])
    return normal / np.linalg.norm(normal)

with open("capture_2.pkl", "rb") as f:
    (rgbs, pcds) = pickle.load(f)

# smoothen pcds
pcds = [smoothen_pcd(pcd) for pcd in pcds]

show_rgb = False
show_pcd = False

if show_rgb:
    plt.title("RGB Image")
    plt.imshow(rgbs[0])
    plt.axis('off')
    plt.show()


# correct for calibration error [duct tape fix; should create more robust solution]
pcds[0][..., 0] += 0.05
pcds[0][..., 2] += 0.015
pcds[1][..., 2] += 0.015

# Select nearby points from the other point cloud.
# target_point = tuple(int(x) for x in input("target: ").split())
# center_xyz = find_valid_point(pcds[0], target_point)
center_xyz = find_valid_point(pcds[0], (667, 511))

# Select points within a certain radius of the median.
# Essentially, we segment the point cloud.
# To do this more accurately, at some point we want to use SAM.
radius = 0.1
all_points = np.concatenate([pcds[0].reshape(-1, 3), pcds[1].reshape(-1, 3)])
all_colors = np.concatenate([rgbs[0].reshape(-1, 3), rgbs[1].reshape(-1, 3)])
distances = np.linalg.norm(all_points - center_xyz, axis=-1)
mask = distances < radius
object_points = all_points[mask]
object_point_colors = all_colors[mask]

# Center the object.
object_points -= center_xyz

if show_pcd:
    fig = plt.figure()
    ax: plt.Axes = fig.add_subplot(projection='3d')
    ax.scatter(object_points[:, 0], object_points[:, 1], object_points[:, 2], c=object_point_colors / 255, s=0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    set_axes_equal(ax)
    plt.show()

lower_bound, upper_bound = np.min(object_points, axis=0), np.max(object_points, axis=0)
voxel_size = 0.005

# Voxelize the point cloud.
voxelized = voxelize(object_points, object_point_colors, (lower_bound, upper_bound), voxel_size)

table_z_index = int((-center_xyz[2] - lower_bound[2]) / voxel_size)
voxelized[:, :, table_z_index] = [0, 0, 0, 1]

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

# Generate grasp candidates.
# We can conjugate the problem into one in which we only care about eef translation.
# For example, rotating the object, revoxelizing, and checking for good grasp candidates.
# We can add the table as a voxel layer manually.
# For now, maybe we just try rotation along the principal axes.

# rotated_voxels = voxel_occupancy.transpose(2, 1, 0)[::-1, :, :]
# rotated_voxel_color = voxel_color.transpose(2, 1, 0, 3)[::-1, :, :]

# fig = plt.figure()
# ax: plt.Axes = fig.add_subplot(projection='3d')
# ax.voxels(rotated_voxels, facecolors=rotated_voxel_color, edgecolor=(1, 1, 1, 0.1))
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# set_axes_equal(ax)
# plt.title("Voxelized Point Cloud")
# plt.show()

# we create several windows over the point cloud
# then for each window we find the minimum and maximum z-values; this tells us the contact point
# additionally we calculate the normal vector at those voxels
# finally, we can check if the grasp is force-closure by looking at the friction cone
# voxelization is just to reduce the number of points in our point cloud to save processing
ws = int(0.01 / voxel_size + 0.5) # round up
h = 2

print(ws, h)

max_y = np.zeros((voxel_occupancy.shape[0], voxel_occupancy.shape[2])) - 1
min_y = np.zeros((voxel_occupancy.shape[0], voxel_occupancy.shape[2])) + 100
gripper_size = 0.15

for y in range(voxel_occupancy.shape[1]):
    mask = voxel_occupancy[:, y, :] > 0
    if not np.any(mask):
        continue
    max_y[mask] = np.maximum(max_y[mask], y)
    min_y[mask] = np.minimum(min_y[mask], y)

grasp_locations = []

for wx in range(ws, voxel_occupancy.shape[0] - ws, h):
    for wz in range(ws, voxel_occupancy.shape[2] - ws, h):
        min_y_window = min_y[wx - ws:wx + ws, wz - ws:wz + ws]
        max_y_window = max_y[wx - ws:wx + ws, wz - ws:wz + ws]
        ymin = np.min(min_y_window) - 1
        ymax = np.max(max_y_window) + 1
        
        if ymin == 99 or ymax == 1:
            continue

        # get normal vector at this point.
        # smoothen the window.
        min_y_window = convolve2d(min_y_window, np.ones((3, 3)) / 9, mode='same', boundary='fill', fillvalue=0)
        max_y_window = convolve2d(max_y_window, np.ones((3, 3)) / 9, mode='same', boundary='fill', fillvalue=0)

        lower_norm = get_normal(min_y_window[ws - 1:ws + 2, ws - 1:ws + 2])
        upper_norm = get_normal(max_y_window[ws - 1:ws + 2, ws - 1:ws + 2])
        # contact direction is vertical
        alpha_lower = np.degrees(np.arccos(lower_norm[2]))
        alpha_upper = np.degrees(np.arccos(upper_norm[2]))
        alpha_lower = min(alpha_lower, 180 - alpha_lower)
        alpha_upper = min(alpha_upper, 180 - alpha_upper)
        
        if wx == 16 and wz == 10:
            print(alpha_lower, alpha_upper)

        # then, calculate alpha
        # finally, see if it's inside or outside the friction cone
        # will just assume that if |alpha| < 15deg, we're fine
        if np.abs(alpha_lower) < 15 and np.abs(alpha_upper) < 15 and (ymax - ymin) * voxel_size < gripper_size:
            print(f"Grasp at ({wx}, {wz})")
            grasp_locations.append((wx, wz, ymin, ymax))

fig = plt.figure()
ax: plt.Axes = fig.add_subplot(projection='3d')
voxel_color[..., -1] = 0.2 # make slightly tranparent
ax.voxels(voxel_occupancy, facecolors=voxel_color, edgecolor=(1, 1, 1, 0.1))
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

# plot grasp locations
for (wx, wz, ymin, ymax) in grasp_locations:
    ax.scatter(wx, ymin, wz, c='r')
    ax.scatter(wx, ymax, wz, c='g')
    ax.plot([wx, wx], [ymin, ymax], [wz, wz], c='b')

set_axes_equal(ax)
plt.title("Voxelized Point Cloud")
plt.show()
