from google.colab import files
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Upload image
uploaded = files.upload()
file_name = next(iter(uploaded))

# 2. Read image in color and convert to grayscale
img_color = cv2.imread(file_name)
gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

# 3. Canny edge detection before smoothing
edges_before = cv2.Canny(gray, 100, 200)

# 4. Gaussian smoothing
smooth = cv2.GaussianBlur(gray, (5, 5), 0)

# 5. Canny edge detection after smoothing
edges_after = cv2.Canny(smooth, 100, 200)

# 6. Display all results
plt.figure(figsize=(12, 8))

# Original Color Image
plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
plt.title("Original Color Image", fontsize=12, fontweight="bold")
plt.axis("off")

# Edges Before Smoothing
plt.subplot(2, 2, 2)
plt.imshow(edges_before, cmap="gray")
plt.title("Edges Before Smoothing (Canny)",
          fontsize=12, fontweight="bold")
plt.axis("off")

# Gaussian Smoothed Image
plt.subplot(2, 2, 3)
plt.imshow(smooth, cmap="gray")
plt.title("Gaussian Smoothed Image (5×5)",
          fontsize=12, fontweight="bold")
plt.axis("off")

# Edges After Smoothing
plt.subplot(2, 2, 4)
plt.imshow(edges_after, cmap="gray")
plt.title("Edges After Smoothing (Canny)",
          fontsize=12, fontweight="bold")
plt.axis("off")

plt.tight_layout()
plt.show()
