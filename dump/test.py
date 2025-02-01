import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt

FOLDER = "./input/"
file_name = f"{FOLDER}{sorted(os.listdir(FOLDER))[4]}"
img = cv.imread(file_name)


img_array = img / 255
print(img_array)

print(f"Max {img_array.max()}")
print(f"Min {img_array.min()}")

img_gray = img_array @ [0.2126, 0.7152, 0.0722]

fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(12, 8))
ax[0] = plt.imshow(img)
ax[1] = plt.imshow(img_gray, cmap="gray")
plt.show()
