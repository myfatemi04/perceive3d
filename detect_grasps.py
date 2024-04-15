import numpy as np
from scipy.signal import convolve2d
import cv2

def get_normal(window):
    # we use finite differences method.
    # window is a 3x3 matrix.
    dx = (window[2, 1] - window[0, 1]) / 2
    dy = (window[1, 2] - window[1, 0]) / 2
    normal = np.array([dx, dy, 1])
    return normal / np.linalg.norm(normal)

def smoothen_window(window):
    # median blur.
    # alternative: SMA.
    # ^^ convolve2d(window_min_z, np.ones((3, 3)) / 9, mode='same', boundary='symm')
    return cv2.medianBlur(window.astype(np.float32), 3) # type: ignore

def detect_grasps(voxel_occupancy, voxel_size, gripper_width, max_alpha, h=2, ws=2):
    """
    Detects vertical grasps in a voxel grid, assuming a parallel-jaw gripper.
    Uses z as the gripper direction.

    we create several windows over the point cloud
    then for each window we find the minimum and maximum z-values; this tells us the contact point
    additionally we calculate the normal vector at those voxels
    finally, we can check if the grasp is force-closure by looking at the friction cone
    voxelization is just to reduce the number of points in our point cloud to save processing
    """
    max_z = np.zeros((voxel_occupancy.shape[0], voxel_occupancy.shape[1])) - 1
    min_z = np.zeros((voxel_occupancy.shape[0], voxel_occupancy.shape[1])) + 10000

    for z in range(voxel_occupancy.shape[2]):
        mask = voxel_occupancy[:, :, z] > 0
        if not np.any(mask):
            continue
        max_z[mask] = np.maximum(max_z[mask], z)
        min_z[mask] = np.minimum(min_z[mask], z)

    grasp_locations = []

    for x in range(ws, voxel_occupancy.shape[0] - ws, h):
        for y in range(ws, voxel_occupancy.shape[1] - ws, h):
            window_min_z = min_z[x - ws:x + ws, y - ws:y + ws]
            window_max_z = max_z[x - ws:x + ws, y - ws:y + ws]
            zmin = np.min(window_min_z) - 1
            zmax = np.max(window_max_z) + 1

            if np.max(window_min_z) == 10000 or np.min(window_max_z) == -1:
                continue

            # get normal vector at this point.
            # smoothen the window.
            window_min_z = smoothen_window(window_min_z)
            window_max_z = smoothen_window(window_max_z)

            # with this smoothed window, calculate the normal (which is the finite differences of the z-values)
            lower_norm = get_normal(window_min_z[ws - 1:ws + 2, ws - 1:ws + 2])
            upper_norm = get_normal(window_max_z[ws - 1:ws + 2, ws - 1:ws + 2])

            # contact direction is vertical
            alpha_lower = np.degrees(np.arccos(lower_norm[2]))
            alpha_upper = np.degrees(np.arccos(upper_norm[2]))
            alpha_lower = min(alpha_lower, 180 - alpha_lower)
            alpha_upper = min(alpha_upper, 180 - alpha_upper)

            # if x == 16 and y == 8:
            #     print(alpha_lower, alpha_upper)
            #     print(lower_norm, upper_norm)
            #     print(window_min_z)
            #     print(window_max_z)

            # then, calculate alpha
            # finally, see if it's inside or outside the friction cone
            # will just assume that if |alpha| < 15deg, we're fine
            contact_distance = (zmax - zmin) * voxel_size
            if max(np.abs(alpha_lower), np.abs(alpha_upper)) < max_alpha and contact_distance < gripper_width:
                grasp_locations.append((x, y, zmin, zmax, alpha_lower, alpha_upper))

    return grasp_locations

