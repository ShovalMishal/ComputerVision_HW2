"""Stereo matching."""
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d
import scipy


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
        kernel = np.ones([win_size, win_size])
        for dis_val_ind, dis_val in enumerate(disparity_values):
            translation_matrix = np.float32([[1, 0, -dis_val], [0, 1, 0]])
            shifted_image = cv2.warpAffine(
                right_image, translation_matrix,
                (right_image.shape[1], right_image.shape[0]))
            ssd = np.sum((left_image - shifted_image) ** 2, axis=2)
            filtered_image = convolve2d(ssd, kernel, mode='same')
            ssdd_tensor[:, :, dis_val_ind] = filtered_image

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
            l_slice[1:-1, col] = c_slice[1:-1, col] + \
                                 np.min(
                                     np.stack([
                                         l_slice[1:-1, col - 1],
                                         p1 + l_slice[:-2, col - 1],
                                         p1 + l_slice[2:, col - 1],
                                         p2 + np.repeat(np.min(l_slice[:, col - 1]), l_slice.shape[0] - 2)], axis=1),
                                     axis=1)
            l_slice[0, col] = c_slice[0, col] + \
                              min(l_slice[0, col - 1],
                                  p1 + l_slice[1, col - 1],
                                  p2 + np.min(l_slice[:, col - 1]))
            l_slice[-1, col] = c_slice[-1, col] + \
                               min(l_slice[-1, col - 1],
                                   p1 + l_slice[-2, col - 1],
                                   p2 + np.min(l_slice[:, col - 1]))
        return l_slice

    @staticmethod
    def dp_grade_slice_old(c_slice: np.ndarray, p1: float, p2: float) -> np.ndarray:
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
                temp_col = np.copy(l_slice[:, col - 1])
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
        for row in range(ssdd_tensor.shape[0]):
            l[row] = self.dp_grade_slice(ssdd_tensor[row].T, p1, p2).T
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
        direction_to_slice = {}
        for direct in range(1, num_of_directions + 1):
            direct_slices = self.extract_slices_acord_direct(ssdd_tensor, direct)
            grade_slices = []
            for direct_slice in direct_slices:
                grade_slices.append(self.dp_grade_slice(direct_slice, p1, p2))
            l = self.aggregate_grade_slices(grade_slices, direct, ssdd_tensor.shape)
            direction_to_slice[direct] = self.naive_labeling(l)
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
        total_l = np.zeros_like(ssdd_tensor)
        for direct in range(1, num_of_directions + 1):
            direct_slices = self.extract_slices_acord_direct(ssdd_tensor, direct)
            grade_slices = []
            for direct_slice in direct_slices:
                grade_slices.append(self.dp_grade_slice(direct_slice, p1, p2))
            total_l += self.aggregate_grade_slices(grade_slices, direct, ssdd_tensor.shape)
        total_l /= num_of_directions
        return self.naive_labeling(total_l)

    @staticmethod
    def extract_slices_acord_direct(ssdd_tensor: np.ndarray, direction: int):
        """
        function which extract slices from the ssdd according to a direction which it receives as an input
        """
        rows, cols = ssdd_tensor.shape[0:2]
        res_slices = []
        if direction == 1 or direction == 5:
            res_slices = ssdd_tensor.transpose((0, 2, 1))
        if direction == 3 or direction == 7:
            res_slices = ssdd_tensor.transpose((1, 2, 0))
        if direction == 5 or direction == 7:
            res_slices = res_slices[:, :, ::-1]
        if direction == 2 or direction == 6 or direction == 8 or direction == 4:
            if direction == 8 or direction == 4:
                ssdd_tensor = ssdd_tensor[::-1, :, :]
            for i in range(-rows + 1, cols):
                diag = ssdd_tensor.diagonal(i)
                if direction == 6 or direction == 4:
                    diag = diag[:, ::-1]
                res_slices.append(diag)

        return res_slices

    @staticmethod
    def aggregate_grade_slices(grade_slices, direct, ssdd_shape) -> np.ndarray:
        rows, cols = ssdd_shape[0:2]
        if direct == 1 or direct == 5 or direct == 3 or direct == 7:
            grade_slices = np.stack(grade_slices)
        if direct == 5 or direct == 7:
            grade_slices = grade_slices[:, :, ::-1]
        if direct == 1 or direct == 5:
            l = grade_slices.transpose((0, 2, 1))
        if direct == 3 or direct == 7:
            l = grade_slices.transpose((2, 0, 1))

        if direct == 2 or direct == 6 or direct == 8 or direct == 4:
            l = np.zeros(ssdd_shape)
            diag_indices = Solution.get_diagonal_indices(rows, cols)
            for ii, curr_diag_indices in enumerate(diag_indices):
                curr_slice = grade_slices[ii]
                if direct == 6 or direct == 4:
                    curr_slice = curr_slice[:, ::-1]
                l[curr_diag_indices] = curr_slice.T
            if direct == 8 or direct == 4:
                l = l[::-1, :, :]
        return l


    @staticmethod
    def get_diagonal_indices(rows, cols):
        """
        find the diagonal indices from offset=-row to offset=col-1
        """
        indices = []
        for row in range(rows):
            diag_inds = (np.arange(row, row + min(rows - row, cols), dtype=np.int16),
                         np.arange(min(rows - row, cols), dtype=np.int16))
            indices.append(diag_inds)
        indices = indices[::-1]
        for col in range(1, cols):
            diag_inds = (np.arange(min(cols - col, rows), dtype=np.int16),
                         np.arange(col, col + min(cols - col, rows), dtype=np.int16))
            indices.append(diag_inds)
        return indices
