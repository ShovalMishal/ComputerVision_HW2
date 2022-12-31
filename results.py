import numpy as np
from matplotlib import pyplot as plt


def compute_forward_mapping(left_img, right_image,label_map, disp_range, ssdd):
    rows, cols = left_img.shape[0], left_img.shape[1]
    result_img = np.zeros_like(left_img)
    min_dis_map = np.inf * np.ones((rows,cols))
    for row in range(rows):
        for col in range(cols):
            new_col = (col + label_map[row, col] - disp_range) % cols
            if ssdd[row, col, label_map[row, col]] < min_dis_map[row, new_col]:
                result_img[row,new_col,:] = left_img[row,col,:]
                min_dis_map[row, new_col]=ssdd[row, col, label_map[row, col]]

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(right_image)
    plt.title('Right Image')
    plt.subplot(1, 2, 2)
    plt.imshow(result_img)
    plt.title('Forward Mapped Left Image')
    plt.show()
