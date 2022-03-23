import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import threshold_otsu

#https://stackoverflow.com/questions/67823386/how-to-find-the-empty-squares-in-a-chess-board-image

# Round to next smaller multiple of 8
# https://www.geeksforgeeks.org/round-to-next-smaller-multiple-of-8/
def round_down_to_next_multiple_of_8(a):
    return a & (-8)

cap = cv2.VideoCapture('Video\\test1639658932.91.avi')
# Read image, and shrink to quadratic shape with width and height of
# next smaller multiple of 8
while(cap.isOpened()):
    ret, frame = cap.read()   
    wh = np.min(round_down_to_next_multiple_of_8(np.array(frame.shape[:2])))
    img = cv2.resize(frame, (wh, wh))

# Prepare some visualization output
    out = img.copy()
    plt.figure(1, figsize=(18, 6))
    plt.subplot(1, 3, 1), plt.imshow(img)

# Blur image
    img = cv2.blur(img, (5, 5))

# Iterate tiles, and count unique colors inside
# https://stackoverflow.com/a/56606457/11089932
    wh_t = wh // 8
    count_unique_colors = np.zeros((8, 8))
    for x in np.arange(8):
        for y in np.arange(8):
            tile = img[y*wh_t:(y+1)*wh_t, x*wh_t:(x+1)*wh_t]
            tile = tile[3:-3, 3:-3]
            count_unique_colors[y, x] = np.unique(tile.reshape(-1, tile.shape[-1]), axis=0).shape[0]

    # Mask empty squares using cutoff from Otsu's method
    val = threshold_otsu(count_unique_colors)
    mask = count_unique_colors < val

    # Some more visualization output
    for x in np.arange(8):
        for y in np.arange(8):
            if mask[y, x]:
                cv2.rectangle(out, (x*wh_t+3, y*wh_t+3),
                          ((x+1)*wh_t-3, (y+1)*wh_t-3), (0, 255, 0), 2)
    plt.subplot(1, 3, 2), plt.imshow(count_unique_colors, cmap='gray')
    plt.subplot(1, 3, 3), plt.imshow(out)
    plt.tight_layout(), plt.show()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break