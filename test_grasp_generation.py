import pickle
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image
from detect_grasps import detect_grasps
from set_axes_equal import set_axes_equal
from voxelize import voxelize


def find_valid_point(pcd, xy, r=10):
    roi_xyz = pcd[
        xy[1] - r : xy[1] + r,
        xy[0] - r : xy[0] + r
    ]
    mask = ~((roi_xyz == -10000).any(axis=-1))
    print(roi_xyz[mask])
    return np.median(roi_xyz[mask], axis=0)

def smoothen_pcd(pcd):
    pcd_smooth = cv2.GaussianBlur(pcd, (5, 5), 2) # type: ignore
    pcd_smooth[pcd == -10000] = -10000
    return pcd_smooth

capture_num = 1
with open(f"capture_{capture_num}.pkl", "rb") as f:
    (rgbs, pcds) = pickle.load(f)

for i in range(2):
    Image.fromarray(rgbs[i]).save(f"capture_{capture_num}_rgb_{i}.png")

# smoothen pcds
pcds = [smoothen_pcd(pcd) for pcd in pcds]

show_rgb = True
show_pcd = True
show_rotated_voxel_clouds = False

if show_rgb:
    plt.title("RGB Image")
    plt.imshow(rgbs[0])
    plt.axis('off')
    plt.show()

# correct for calibration error [duct tape fix; should create more robust solution]
mask0 = pcds[0] == -10000
mask1 = pcds[1] == -10000
pcds[0][..., 0] += 0.05
pcds[0][..., 2] += 0.015
pcds[1][..., 2] += 0.015
pcds[0][mask0] = -10000
pcds[1][mask1] = -10000

# Select nearby points from the other point cloud.
if show_rgb:
    target_point = tuple(int(x) for x in input("target: ").split())
    center_xyz = find_valid_point(pcds[0], target_point)
else:
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
    ax.scatter(object_points[:, 0], object_points[:, 1], object_points[:, 2], c=object_point_colors / 255, s=0.5) # type: ignore
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z') # type: ignore
    set_axes_equal(ax)
    plt.show()


lower_bound, upper_bound = np.min(object_points, axis=0), np.max(object_points, axis=0)
voxel_size = 0.005
min_points_in_voxel = 2

# Voxelize the point cloud.
voxelized = voxelize(object_points, object_point_colors, (lower_bound, upper_bound), voxel_size)

# table_z_index = int((-center_xyz[2] - lower_bound[2]) / voxel_size)
table_z_index = 0
voxelized[:, :, table_z_index] = [0, 0, 0, 1]

voxel_occupancy = np.zeros(voxelized.shape[:-1], dtype=bool)
voxel_occupancy[voxelized[..., -1] >= min_points_in_voxel] = True
voxel_color = voxelized.copy()
voxel_color[..., -1] = 1.0

fig = plt.figure()
ax: plt.Axes = fig.add_subplot(projection='3d')
ax.voxels(voxel_occupancy, facecolors=voxel_color, edgecolor=(1, 1, 1, 0.1)) # type: ignore
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z') # type: ignore
set_axes_equal(ax)
plt.title("Voxelized Point Cloud")
plt.show()

# Generate grasp candidates.
# We can conjugate the problem into one in which we only care about eef translation.
# For example, rotating the object, revoxelizing, and checking for good grasp candidates.
# We can add the table as a voxel layer manually.
# For now, maybe we just try rotation along the principal axes.

mask = object_points[..., 2] < -center_xyz[2]
object_points = object_points[~mask]
object_point_colors = object_point_colors[~mask]

grasps = []

start_time = time.time()

for i in range(8):
    rotate_angle = np.pi / 8 * i
    z_inv = np.array([np.cos(rotate_angle), np.sin(rotate_angle), 0])
    x_inv = np.array([np.cos(rotate_angle - np.pi/2), np.sin(rotate_angle - np.pi/2), 0])
    y_inv = np.cross(z_inv, x_inv)
    rotation_matrix = np.array([x_inv, y_inv, z_inv])

    # apply rotation matrix to points
    rotated_object_points = object_points @ rotation_matrix.T
    lower_bound_, upper_bound_ = np.min(rotated_object_points, axis=0), np.max(rotated_object_points, axis=0)
    voxels_ = voxelize(rotated_object_points, object_point_colors, (lower_bound_, upper_bound_), voxel_size)

    ws = 2 # int(0.01 / voxel_size + 0.5) # round up
    h = 1 # 2
    gripper_width = 0.2

    voxel_occupancy_ = (voxels_[:, :, :, -1] >= min_points_in_voxel)
    max_alpha = 15
    grasps_voxelized = detect_grasps(voxel_occupancy_, voxel_size, gripper_width, max_alpha, h, ws)
    # translate these into the original frame.
    # these are in (x, y, zmin, zmax) format.
    grasps_from_this = []
    for (x, y, zmin, zmax, alpha_lower, alpha_upper) in grasps_voxelized:
        start = (np.array([x, y, zmin]) * voxel_size + lower_bound_) @ rotation_matrix
        end = (np.array([x, y, zmax]) * voxel_size + lower_bound_) @ rotation_matrix
        worst_alpha = max(abs(alpha_lower), abs(alpha_upper))
        grasps_from_this.append((worst_alpha, start, end))

    # select top grasps by force closure
    grasps_from_this.sort(key=lambda x: x[0])

    grasps.extend([(start, end) for (_, start, end) in grasps_from_this[:5]])

    if True:# show_rotated_voxel_clouds:
        fig = plt.figure()
        ax: plt.Axes = fig.add_subplot(projection='3d')
        ax.set_title(f"Rotation Angle: $\\frac{{{i}\\pi}}{{8}}$")

        voxel_color_ = voxels_.copy()
        voxel_color_[..., -1] = 1.0
        ax.voxels(voxel_occupancy_, facecolors=voxel_color_, edgecolor=(1, 1, 1, 0.1)) # type: ignore
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z') # type: ignore
        set_axes_equal(ax)
        plt.show()

end_time = time.time()

print("Time:", end_time - start_time)
print("Hz:", 1/(end_time - start_time))

fig = plt.figure()
ax: plt.Axes = fig.add_subplot(projection='3d')
voxel_color[..., -1] = 0.1 # make slightly tranparent

ax.voxels(voxel_occupancy, facecolors=voxel_color, edgecolor=(1, 1, 1, 0.1))  # type: ignore
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z') # type: ignore

# plot grasp locations
for grasp in grasps:
    ((x1, y1, z1), (x2, y2, z2)) = grasp
    x1 -= lower_bound[0]
    y1 -= lower_bound[1]
    z1 -= lower_bound[2]
    x2 -= lower_bound[0]
    y2 -= lower_bound[1]
    z2 -= lower_bound[2]
    x1 /= voxel_size
    y1 /= voxel_size
    z1 /= voxel_size
    x2 /= voxel_size
    y2 /= voxel_size
    z2 /= voxel_size
    ax.scatter(x1, y1, z2, c='r')
    ax.scatter(x2, y2, z2, c='g')
    ax.plot([x1, x2], [y1, y2], [z2, z2], c='b')

set_axes_equal(ax)
plt.title("Voxelized Point Cloud")
plt.show()
