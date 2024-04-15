"""
Segment point clouds using SAM.
"""

from typing import List

from matplotlib import pyplot as plt
import numpy as np
import PIL.Image as Image
from transformers import SamModel, SamProcessor


class SamPointCloudSegmenter():
    def __init__(self, device='cpu', render_2d_results=False):
        self.model: SamModel = SamModel.from_pretrained("facebook/sam-vit-base") # type: ignore
        self.processor: SamProcessor = SamProcessor.from_pretrained("facebook/sam-vit-base") # type: ignore
        self.render_2d_results = render_2d_results
        self.device = device

    def _segment_image(self, image: Image.Image, input_points=None, input_boxes=None):
        inputs = self.processor(images=[image], input_points=input_points, input_boxes=input_boxes, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        masks = self.processor.image_processor.post_process_masks( # type: ignore
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),      # type: ignore
            inputs["reshaped_input_sizes"].cpu() # type: ignore
        )
        scores = outputs.iou_scores

        # Render intermediate segmentation result
        if self.render_2d_results:
            plt.imshow(image) # type: ignore # It gets converted to a Numpy array.
            if input_points is not None:
                x = [point[0] for point in input_points]
                y = [point[1] for point in input_points]
                plt.scatter(x, y, color='red', label='Input points')
            elif input_boxes is not None:
                for input_box in input_boxes:
                    x1, y1, x2, y2 = input_box
                    plt.plot([x1, x1, x2, x2, x1], [y1, y2, y2, y1, y1], color='blue', label='Input box')
            
            for mask in masks:
                plt.imshow(mask, alpha=0.5)
            plt.show()

        return (masks, scores)

    def transfer_segmentation(self,
                              segmentation: np.ndarray,
                              base_point_cloud: np.ndarray,
                              supplementary_rgb_image: Image.Image,
                              supplementary_point_cloud: np.ndarray,
                              max_distance=0.02):
        """
        Takes a segmentation mask for a given point cloud, and generates a prompt to segment another point cloud.
        """
        segmented_points: np.ndarray = base_point_cloud[segmentation]
        valid_mask = ~(segmented_points == -10000).any(axis=-1)
        segmented_points = segmented_points[valid_mask] # (n1, 3)

        # Finds nearby points in the supplementary image.
        n1 = segmented_points.shape[0]
        # Filter out invalid points in the supplementary image.
        w, h = supplementary_rgb_image.size
        coordinates = np.zeros((h, w, 2))
        coordinates[..., 0] = np.arange(w)[None, :].repeat(h, w)
        coordinates[..., 1] = np.arange(h)[:, None].repeat(h, w)
        valid_mask = ~(supplementary_point_cloud == -10000).any(axis=-1)
        supplementary_point_cloud = supplementary_point_cloud[valid_mask] # (n2, 4)
        coordinates = coordinates[valid_mask]
        n2 = supplementary_point_cloud.shape[0]
        # Calculates distance between all pairs of points.
        # Could definitely be optimized. Matrix shape: (n1, n2)
        dists = np.linalg.norm(segmented_points[:, None, 3].repeat((1, n2, 1)) - supplementary_point_cloud[None, :, 3].repeat((n1, 1, 1)), axis=-1)
        # Get the right coordinates
        coordinates[(dists < max_distance).any(axis=0)]
        # Construct a bounding box
        x1 = np.min(coordinates[:, 0])
        y1 = np.min(coordinates[:, 1])
        x2 = np.max(coordinates[:, 0])
        y2 = np.max(coordinates[:, 1])

        transferred_segmentation = self._segment_image(supplementary_rgb_image, input_boxes=[[x1, y1, x2, y2]])

        return transferred_segmentation

    def segment(self, base_rgb_image: Image.Image, base_point_cloud: np.ndarray, prompt_bounding_box: np.ndarray, supplementary_rgb_images: List[Image.Image], supplementary_point_clouds: List[np.ndarray]):
        """
        Given a base RGB + point cloud image, and a prompt for that image,
        segment the point cloud using the SAM model. Then, fill out the point
        cloud using the other images.
        """

        base_segmentation_masks, base_segmentation_scores = self._segment_image(base_rgb_image, input_boxes=[prompt_bounding_box])
        base_segmentation_cloud = base_point_cloud[base_segmentation_masks[0]]

        point_clouds = [base_segmentation_cloud]

        # Transfer the segmentation to the other point clouds.
        for supplementary_rgb_image, supplementary_point_cloud in zip(supplementary_rgb_images, supplementary_point_clouds):
            (transferred_segmentation_masks, transferred_segmentation_scores) = \
                self.transfer_segmentation(base_segmentation_masks[0], base_point_cloud, supplementary_rgb_image, supplementary_point_cloud)
            
            # Add the resulting points.
            point_clouds.append(supplementary_point_cloud[transferred_segmentation_masks[0]])

        point_cloud = np.concatenate(point_clouds).reshape(-1, 3)
        valid = ~(point_cloud == -10000).any(axis=-1)
        point_cloud = point_cloud[valid]

        return np.ascontiguousarray(point_cloud)
