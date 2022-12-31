"""Stereo matching."""
import numpy as np
from scipy.signal import convolve2d


class Solution:
    def __init__(self):
        pass

    @staticmethod
    def ssd_distance(left_image: np.ndarray,
                     right_image: np.ndarray,
                     win_size: int,
                     dsp_range: int) -> np.ndarray:
        """Compute the SSDD distances tensor.

        Args:
            left_image: Left image of shape: HxWx3, and type np.double64.
            right_image: Right image of shape: HxWx3, and type np.double64.
            win_size: Window size odd integer.
            dsp_range: Half of the disparity range. The actual range is
            -dsp_range, -dsp_range + 1, ..., 0, 1, ..., dsp_range.

        Returns:
            A tensor of the sum of squared differences for every pixel in a
            window of size win_size X win_size, for the 2*dsp_range + 1
            possible disparity values. The tensor shape should be:
            HxWx(2*dsp_range+1).
        """
        num_of_rows, num_of_cols = left_image.shape[0], left_image.shape[1]
        disparity_values = range(-dsp_range, dsp_range + 1)
        ssdd_tensor = np.zeros((num_of_rows,
                                num_of_cols,
                                len(disparity_values)))
        third_dim = left_image.shape[2]
        padded_left_image = np.zeros(
            (num_of_rows + 2 * (win_size // 2), num_of_cols + 2 * (win_size // 2), third_dim))
        padded_right_image = np.zeros(
            (num_of_rows + 2 * (win_size // 2), num_of_cols + 2 * (win_size // 2), third_dim))
        padded_left_image[win_size // 2:num_of_rows + win_size // 2,
        win_size // 2:num_of_cols + win_size // 2] = left_image
        padded_right_image[win_size // 2:num_of_rows + win_size // 2,
        win_size // 2:num_of_cols + win_size // 2] = right_image
        for row in range(num_of_rows):
            for col in range(num_of_cols):
                for dis_val in disparity_values:
                    curr_pixel_row = row + win_size // 2
                    curr_pixel_col = col + win_size // 2
                    left_window = padded_left_image[curr_pixel_row - win_size // 2: curr_pixel_row + win_size // 2 + 1,
                                  curr_pixel_col - win_size // 2: curr_pixel_col + win_size // 2 + 1]
                    new_col_index = (col + dis_val) % num_of_cols + win_size // 2
                    right_window = padded_right_image[
                                   curr_pixel_row - win_size // 2: curr_pixel_row + win_size // 2 + 1,
                                   new_col_index - win_size // 2: new_col_index + win_size // 2 + 1]
                    ssdd_tensor[row, col, dis_val + dsp_range] = np.sum((left_window - right_window) ** 2)

        ssdd_tensor -= ssdd_tensor.min()
        ssdd_tensor /= ssdd_tensor.max()
        ssdd_tensor *= 255.0
        return ssdd_tensor

    @staticmethod
    def naive_labeling(ssdd_tensor: np.ndarray) -> np.ndarray:
        """Estimate a naive depth estimation from the SSDD tensor.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for every
            pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.

        Evaluate the labels in a naive approach. Each value in the
        result tensor should contain the disparity matching minimal ssd (sum of
        squared difference).

        Returns:
            Naive labels HxW matrix.
        """
        label_no_smooth = np.argmin(ssdd_tensor, axis=2)
        return label_no_smooth

    @staticmethod
    def dp_grade_slice(c_slice: np.ndarray, p1: float, p2: float) -> np.ndarray:
        """Calculate the scores matrix for slice c_slice.

        Calculate the scores slice which for each column and disparity value
        states the score of the best route. The scores slice is of shape:
        (2*dsp_range + 1)xW.

        Args:
            c_slice: A slice of the ssdd tensor.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.
        Returns:
            Scores slice which for each column and disparity value states the
            score of the best route.
        """
        num_labels, num_of_cols = c_slice.shape[0], c_slice.shape[1]
        l_slice = np.zeros((num_labels, num_of_cols))
        l_slice[:, 0] = c_slice[:, 0]
        labels_indices = np.arange(num_labels)
        for col in range(1, num_of_cols):
            term_a = l_slice[:, col - 1]
            term_b = p1 + np.minimum(l_slice[(labels_indices - 1) % num_labels, col - 1],
                                     l_slice[(labels_indices + 1) % num_labels, col - 1])
            term_c = np.zeros_like(term_b)
            for d in range(0, num_labels):
                temp_col = l_slice[:, col - 1]
                temp_col[d] = np.inf
                temp_col[(d - 1) % num_labels] = np.inf
                temp_col[(d + 1) % num_labels] = np.inf
                min_los = np.min(temp_col)
                term_c[d] = min_los
            term_c += p2
            M_col = np.minimum(np.minimum(term_a, term_b), term_c)
            l_slice[:, col] = c_slice[:, col] + M_col - np.min(l_slice[:, col - 1])
        return l_slice

    def dp_labeling(self,
                    ssdd_tensor: np.ndarray,
                    p1: float,
                    p2: float) -> np.ndarray:
        """Estimate a depth map using Dynamic Programming.

        (1) Call dp_grade_slice on each row slice of the ssdd tensor.
        (2) Store each slice in a corresponding l tensor (of shape as ssdd).
        (3) Finally, for each pixel in l (along each row and column), choose
        the best disparity value. That is the disparity value which
        corresponds to the lowest l value in that pixel.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for every
            pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.
        Returns:
            Dynamic Programming depth estimation matrix of shape HxW.
        """
        l = np.zeros_like(ssdd_tensor)
        num_of_rows = ssdd_tensor.shape[0]
        for row in range(num_of_rows):
            cur_slice = ssdd_tensor[row, :, :].T
            l[row, :, :] = self.dp_grade_slice(cur_slice, p1, p2).T
        return self.naive_labeling(l)

    def dp_labeling_per_direction(self,
                                  ssdd_tensor: np.ndarray,
                                  p1: float,
                                  p2: float) -> dict:
        """Return a dictionary of directions to a Dynamic Programming
        etimation of depth.

        For each direction in 1, ..., 8, calculate scores tensors
        according to dp_grade_slice and the method which allows you to
        extract slices along each direction.

        You may use helper methods (functions) that you write on your own.
        We found `np.diagonal` to be very helpful to extract diagonal slices.
        `np.unravel_index` might be helpful if you're thinking in MATLAB
        notations: it's the ind2sub equivalent.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for
            every pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.

        Returns:
            Dictionary int->np.ndarray which maps each direction to the
            corresponding dynamic programming estimation of depth based on
            that direction.
        """
        num_of_directions = 8
        l = np.zeros_like(ssdd_tensor)
        direction_to_slice = {}
        """INSERT YOUR CODE HERE"""
        return direction_to_slice

    def sgm_labeling(self, ssdd_tensor: np.ndarray, p1: float, p2: float):
        """Estimate the depth map according to the SGM algorithm.

        For each direction in 1, ..., 8, calculate scores tensors
        according to dp_grade_slice and the method which allows you to
        extract slices along each direction.

        You may use helper methods (functions) that you write on your own.
        We found `np.diagonal` to be very helpful to extract diagonal slices.
        `np.unravel_index` might be helpful if you're thinking in MATLAB
        notations: it's the ind2sub equivalent.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for
            every pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.

        Returns:
            Semi-Global Mapping depth estimation matrix of shape HxW.
        """
        num_of_directions = 8
        l = np.zeros_like(ssdd_tensor)
        """INSERT YOUR CODE HERE"""
        return self.naive_labeling(l)
