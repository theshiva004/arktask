
import numpy as np
import cv2
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import requests

def disparity(im1, im2, max_disp=64, win_size=7):
    h, w = im1.shape
    dispM = np.zeros((h, w), dtype=np.float32)
    min_ssd = np.full((h, w), np.inf)

    im1 = im1.astype(np.float32)
    im2 = im2.astype(np.float32)

    kernel = np.ones((win_size, win_size), dtype=np.float32)

    for d in range(max_disp + 1):
        # Shift im2 by d pixels to the right
        shifted_im2 = np.zeros_like(im2)
        if d > 0:
            shifted_im2[:, d:] = im2[:, :-d]
        else:
            shifted_im2 = im2.copy()

        # Compute ssd and sum using convolution
        ssd = (im1 - shifted_im2) ** 2
        ssd_sum = convolve2d(ssd, kernel, mode='same', boundary='fill', fillvalue=0)

        # Update disparity and min_ssd where SSD is lower 
        for y in range(h):
            for x in range(w):
                if ssd_sum[y, x] < min_ssd[y, x]:
                    min_ssd[y, x] = ssd_sum[y, x]
                    dispM[y, x] = d

    return dispM


def normalize_disparity(disparity_map):
    min_val = np.min(disparity_map)
    max_val = np.max(disparity_map)
    normalized = 255 * (disparity_map - min_val) / (max_val - min_val)
    return normalized.astype(np.uint8)

def main():
    response1 = requests.get("https://github.com/theshiva004/arktask/raw/main/left.png", stream=True).raw
    image_array1 = np.asarray(bytearray(response1.read()), dtype=np.uint8)

    left=cv2.imdecode(image_array1, cv2.IMREAD_COLOR) 

    response2 = requests.get("https://github.com/theshiva004/arktask/raw/main/right.png", stream=True).raw
    image_array2 = np.asarray(bytearray(response2.read()), dtype=np.uint8)

    right=cv2.imdecode(image_array2, cv2.IMREAD_COLOR)

    gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    
    
    disparity_map = disparity(gray_left, gray_right)
    normalized_disp = normalize_disparity(disparity_map)
    heatmap = cv2.applyColorMap(disparity_map, cv2.COLORMAP_JET)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(left, cv2.COLOR_BGR2RGB))
    plt.title('Left Image')

    plt.subplot(2, 2, 2)
    plt.imshow(cv2.cvtColor(right, cv2.COLOR_BGR2RGB))
    plt.title('Right Image')
    
    plt.subplot(2, 2, 3)
    plt.imshow(normalized_disp, cmap='gray')
    plt.title('Disparity Map')
    
    plt.subplot(2, 2, 4)
    plt.imshow(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
    plt.title('Depth Heatmap')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
   main()
