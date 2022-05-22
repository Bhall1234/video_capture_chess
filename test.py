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

       """
        for x in range(0,rotated.shape[0] - 8, rotated.shape[0]//8):
            for y in range(0,rotated.shape[1] - 8, rotated.shape[1]//8):      
                square = rotated[x:x+rotated.shape[0]//8, y:y+rotated.shape[1]//8, :]          # creating 8*8 squares of image
                avg_colour_per_row = np.average(square, axis=0)
                avg_colour = np.array(list(map(int, np.average(avg_colour_per_row, axis=0))))//8         # finding average colour of the square
        
                if list(avg_colour) == list(np.array([0, 0, 0])) or list(avg_colour) == list(np.array([31, 31, 31])):         # if average colour  of the squareis black or white, then print the coordinates of the square
                    print(x//(rotated.shape[0]//8), y//(rotated.shape[1]//8))
        
        wh = np.min(round_down_to_next_multiple_of_8(np.array(rotated.shape[:2])))
        img = cv2.resize(rotated, (wh, wh))

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
        """