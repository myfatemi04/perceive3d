import numpy as np

def voxelize(point_cloud, colors, box_bounds, voxel_size):
    """
    Given a point cloud, generate a voxel grid.
    """
    voxelized_point_cloud = ((point_cloud - box_bounds[0]) / voxel_size).astype(int)
    grid_dims = ((box_bounds[1] - box_bounds[0]) / voxel_size).astype(int)

    # Crop anything that is out-of-bounds.
    mask = ((0 <= voxelized_point_cloud) & (voxelized_point_cloud < grid_dims)).all(axis=-1)

    # Count occupancy.
    occupancy = np.zeros(grid_dims, dtype=int)
    for (x, y, z) in voxelized_point_cloud[mask]:
        occupancy[x, y, z] += 1

    # (r, g, b, a)
    voxels = np.zeros((*grid_dims, 4), dtype=float)
    # print(voxelized_point_cloud[mask], grid_dims)
    voxels[..., :-1][tuple(voxelized_point_cloud[mask].T)] = colors[mask] / 255.0
    voxels[..., -1] = occupancy > 1 # / (np.max(occupancy) + 1e-6)

    return voxels

