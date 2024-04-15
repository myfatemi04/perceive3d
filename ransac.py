import numpy as np

def get_bounding_box_ransac(points, min_inliers, n_hypothetical_inliers, max_iterations=10):
    """
    We want to get a bounding box without worrying about a couple points messing us up
    (for example, depth detections from far away, that correspond to misclassified/missegmented
    pixels)
    """

    for iteration in range(max_iterations):
        selection = np.random.uniform(0, 1, size=points.shape[0]).argsort()[:n_hypothetical_inliers]
        selected_points = points[selection]

        # Find bounding box of selected points
        max_pt = np.max(selected_points, axis=0)
        min_pt = np.min(selected_points, axis=0)

        # Find inliers
        inliers = (points >= min_pt) & (points <= max_pt)
        inliers = inliers.all(axis=-1)

        if len(inliers) >= min_inliers:
            return min_pt, max_pt, inliers
        
    return None, None, None
