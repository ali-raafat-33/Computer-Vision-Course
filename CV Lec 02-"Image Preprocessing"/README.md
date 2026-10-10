# Computer Vision — Image Operations & Visualization

In this lecture we learn the basic operations we apply to images **before** feeding them to any model, and how to **look at** images and their data to understand what is going on.

> All examples use one sample image: `images/sample.jpg` (the "astronaut" image from scikit-image, public domain, 512×512).

## Table of Contents

**1.5 Image Operations**
1. [Resize](#1-resize)
2. [Crop](#2-crop)
3. [Rotate](#3-rotate)
4. [Flip](#4-flip)
5. [Normalize](#5-normalize)
6. [Convert Color Spaces](#6-convert-color-spaces)

**1.6 Image Visualization**
1. [Displaying Images](#1-displaying-images)
2. [Displaying Multiple Images](#2-displaying-multiple-images)
3. [Histograms](#3-histograms)
4. [Pixel Inspection](#4-pixel-inspection)

---

## Setup

```bash
pip install opencv-python numpy matplotlib
```

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("images/sample.jpg")   # loaded as a NumPy array
print(img.shape, img.dtype)             # (512, 512, 3) uint8
```

### Important: an image is just a NumPy array

- Shape is `(height, width, channels)`, so the **first index is the row (y)** and the second is the **column (x)**.
- Each value is `uint8`: an integer from `0` (black) to `255` (white).
- **OpenCV loads color images as BGR, not RGB.** Matplotlib expects RGB. If you display a BGR image directly, the colors look wrong (red and blue are swapped).

![BGR vs RGB](images/01_bgr_vs_rgb.png)

```python
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # use this for plt.imshow
```

---

# 1.5 Image Operations

## 1. Resize

**What it is:** changing the width and height of an image.

**Why we need it:** most models require a fixed input size (e.g. 224×224). Smaller images are also faster to process.

**How to choose the interpolation method:** when pixels are created or removed, OpenCV has to estimate new pixel values.

| Method | Best for |
|---|---|
| `cv2.INTER_AREA` | **Shrinking** images (best quality) |
| `cv2.INTER_LINEAR` | General use (default) |
| `cv2.INTER_CUBIC` | **Enlarging** images (smoother, slower) |
| `cv2.INTER_NEAREST` | Fastest, blocky result (keeps hard edges) |

![Resize](images/02_resize.png)

```python
# Resize to an exact size. NOTE: the order is (width, height)
small = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)

# Resize by a scale factor (keeps the aspect ratio)
half = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

# Enlarge: compare two interpolation methods
big_nearest = cv2.resize(small, (512, 512), interpolation=cv2.INTER_NEAREST)  # blocky
big_cubic   = cv2.resize(small, (512, 512), interpolation=cv2.INTER_CUBIC)    # smooth

print(img.shape, "->", small.shape)   # (512, 512, 3) -> (128, 128, 3)
```

> **Common mistake:** `img.shape` is `(height, width)` but `cv2.resize` takes `(width, height)`.

---

## 2. Crop

**What it is:** keeping only a rectangular region of the image (Region of Interest, ROI).

**Why we need it:** remove useless background, focus on the object (a face, a license plate), or cut an image into patches.

**How it works:** there is no special function. Since the image is a NumPy array, cropping is just **slicing**: `img[y1:y2, x1:x2]`.

![Crop](images/03_crop.png)

```python
y1, y2 = 40, 200      # rows    (top to bottom)
x1, x2 = 170, 330     # columns (left to right)

crop = img[y1:y2, x1:x2]
print(crop.shape)     # (160, 160, 3)

# Use .copy() if you want to modify the crop without changing the original
crop = img[y1:y2, x1:x2].copy()
```

> **Remember:** slicing returns a *view*, not a copy. Changing `crop` changes `img` unless you call `.copy()`.

---

## 3. Rotate

**What it is:** turning the image around a point by some angle.

**Why we need it:** fixing wrongly oriented photos, and **data augmentation** (making the model robust to rotated objects).

**Two ways:**
1. `cv2.rotate`: fast, only for 90°, 180°, 270° with no quality loss.
2. `cv2.warpAffine`: any angle, using a rotation matrix. With arbitrary angles the corners get cut off unless we enlarge the output canvas.

![Rotate](images/04_rotate.png)

```python
# 1) Fixed angles
rot90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
# also: cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE

# 2) Any angle
h, w = img.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, angle=45, scale=1.0)   # positive angle = counter-clockwise
rotated = cv2.warpAffine(img, M, (w, h))                   # corners are cut off

# 3) Any angle WITHOUT cutting the corners: enlarge the canvas first
cos, sin = abs(M[0, 0]), abs(M[0, 1])
new_w = int(h * sin + w * cos)
new_h = int(h * cos + w * sin)
M[0, 2] += new_w / 2 - center[0]
M[1, 2] += new_h / 2 - center[1]
rotated_full = cv2.warpAffine(img, M, (new_w, new_h))
```

---

## 4. Flip

**What it is:** mirroring the image.

**Why we need it:** another very common **data augmentation**. A flipped cat is still a cat.

| `flipCode` | Effect |
|---|---|
| `1` | Horizontal (left ↔ right) |
| `0` | Vertical (top ↕ bottom) |
| `-1` | Both |

![Flip](images/05_flip.png)

```python
flip_h    = cv2.flip(img, 1)    # horizontal
flip_v    = cv2.flip(img, 0)    # vertical
flip_both = cv2.flip(img, -1)   # both

# Same thing with NumPy slicing
flip_h_np = img[:, ::-1]        # reverse the columns
flip_v_np = img[::-1, :]        # reverse the rows
```

> **Be careful:** flipping is not always safe. Flipping text or digits (6 vs 9, or letters) changes the meaning.

---

## 5. Normalize

**What it is:** changing the range of pixel values, usually from `[0, 255]` to a smaller range.

**Why we need it:** neural networks train **faster and more stably** when inputs are small and centered around zero. Large values like 255 make gradients unstable.

**Two common types:**
1. **Scaling to [0, 1]:** divide by 255.
2. **Standardization (mean/std):** subtract the mean and divide by the std for each channel. Pretrained models (e.g. ResNet on ImageNet) expect the ImageNet mean/std shown below.

![Normalize](images/06_normalize.png)

The image looks almost the same, but the **numbers** (see the histograms in the second row) have a totally different range.

```python
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 1) Scale to [0, 1]
img_01 = img_rgb.astype(np.float32) / 255.0
print(img_01.min(), img_01.max())          # 0.0 1.0

# 2) Standardize with ImageNet mean/std (per channel, RGB order)
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
img_std = (img_01 - mean) / std
print(img_std.min(), img_std.max())        # roughly -2.1 to 2.6

# 3) Min-Max normalization (stretch to the full [0, 1] range)
img_minmax = cv2.normalize(img_rgb, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
```

> **Common mistake:** convert to `float32` **before** dividing. If you stay in `uint8` the result is wrong. Also, the mean/std order must match the channel order (RGB vs BGR).

---

## 6. Convert Color Spaces

**What it is:** representing the same image with different channels.

**Why we need it:** each color space makes a different task easier.

| Color space | Channels | Useful for |
|---|---|---|
| **RGB / BGR** | Red, Green, Blue | Display, deep learning input |
| **Grayscale** | 1 channel (brightness) | Faster processing, edge detection, when color is not needed |
| **HSV** | Hue, Saturation, Value | **Color-based segmentation** (e.g. "find all red objects"), because color (H) is separated from brightness (V) |
| **LAB** | Lightness, a, b | Color comparison closer to human perception |

![Color spaces](images/07_color_spaces.png)

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # shape (512, 512), a single channel
hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lab  = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

print(img.shape, gray.shape)   # (512, 512, 3) (512, 512)
```

### Splitting channels

Each channel is just a grayscale image showing "how much" of that component each pixel has. Bright = high value.

![Channels](images/08_channels.png)

```python
b, g, r = cv2.split(img)                  # OpenCV order is B, G, R
h, s, v = cv2.split(hsv)

merged = cv2.merge([b, g, r])             # put them back together
```

> **Note on HSV in OpenCV:** Hue goes from `0` to `179` (not 360), while S and V go from `0` to `255`.

---

# 1.6 Image Visualization

## 1. Displaying Images

**Why it matters:** in computer vision you constantly need to *see* the result of every step to catch mistakes early.

**Using Matplotlib** (works in Jupyter and Colab):

```python
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))   # convert BGR -> RGB first
plt.title("Original image")
plt.axis("off")                                     # hide the axes
plt.show()
```

**Grayscale images** have a single channel, so Matplotlib colors them with a default colormap (viridis, greenish) unless you set `cmap="gray"`:

![Colormaps](images/09_display_cmap.png)

```python
plt.imshow(gray, cmap="gray")       # correct for grayscale
plt.show()
```

**Using OpenCV windows** (only in normal Python scripts, **not** Jupyter/Colab):

```python
cv2.imshow("Image", img)    # OpenCV expects BGR, so no conversion needed
cv2.waitKey(0)              # wait for any key
cv2.destroyAllWindows()
```

> In Google Colab use `from google.colab.patches import cv2_imshow` and `cv2_imshow(img)`, or just use Matplotlib.

---

## 2. Displaying Multiple Images

**Why it matters:** comparing "before vs after" is the best way to check that an operation did what you expected.

**How it works:** `plt.subplots(rows, cols)` creates a grid of axes, and we draw one image in each.

![Multiple images](images/10_multiple_images.png)

```python
images = [
    ("Original",    cv2.cvtColor(img, cv2.COLOR_BGR2RGB), None),
    ("Grayscale",   gray,                                  "gray"),
    ("Flipped",     cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB), None),
    ("Cropped",     cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), None),
    ("Rotated 90°", cv2.cvtColor(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), cv2.COLOR_BGR2RGB), None),
    ("Blurred",     cv2.cvtColor(cv2.GaussianBlur(img, (15, 15), 0), cv2.COLOR_BGR2RGB), None),
]

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for ax, (title, image, cmap) in zip(axes.ravel(), images):
    ax.imshow(image, cmap=cmap)
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()
plt.show()
```

> Images can have different sizes in the grid. Matplotlib scales each one to fit its own axes.

---

## 3. Histograms

**What it is:** a chart that counts **how many pixels have each intensity value** (0 to 255).

**How to read it:**
- Peak on the **left** → dark image.
- Peak on the **right** → bright image.
- Values squeezed in a **narrow range** → low contrast.
- Spread over the **whole range** → good contrast.

**Why we need it:** judging exposure and contrast, choosing thresholds, and comparing images.

![Histograms](images/11_histograms.png)

```python
# Grayscale histogram
plt.hist(gray.ravel(), bins=256, range=(0, 256), color="gray")
plt.xlabel("Pixel value")
plt.ylabel("Count")
plt.show()

# Color histogram: one curve per channel
for i, color in enumerate(["b", "g", "r"]):
    hist = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(hist, color=color, label=color.upper())
plt.legend()
plt.show()
```

### Histogram Equalization

Spreads the pixel values over the full range, which **boosts contrast** in dull images. Notice how the histogram becomes wider and flatter:

![Equalization](images/12_equalization.png)

```python
equalized = cv2.equalizeHist(gray)    # works on single-channel (grayscale) images only
```

---

## 4. Pixel Inspection

**What it is:** reading (or changing) the actual values of specific pixels.

**Why we need it:** to *understand* that an image is just numbers, to debug (is the range 0-255 or 0-1? BGR or RGB?), and to check that an operation changed what you expect.

![Pixel inspection](images/13_pixel_inspection.png)

```python
row, col = 100, 250

# Color image: returns all 3 channels (B, G, R)
print(img[row, col])            # [24 43 64]

# A single channel of that pixel
print(img[row, col, 0])         # Blue value -> 24

# Grayscale image: returns one number
print(gray[row, col])           # 47

# Look at a small neighborhood (9x9 patch around the pixel)
patch = gray[row-4:row+5, col-4:col+5]
print(patch)

# Change a pixel (or a region) directly
img_copy = img.copy()
img_copy[100:120, 250:270] = (0, 0, 255)    # paint a red square (BGR)
```

### Useful image statistics

```python
print("Shape :", img.shape)          # (512, 512, 3)
print("Dtype :", img.dtype)          # uint8
print("Min   :", img.min())
print("Max   :", img.max())
print("Mean per channel (B,G,R):", img.mean(axis=(0, 1)))
```

> **Remember:** index order is `[row, col]` = `[y, x]`. This is the opposite of what we usually write for coordinates `(x, y)`.

---

## Summary

| Operation | Function | Typical use |
|---|---|---|
| Resize | `cv2.resize` | Fixed model input size |
| Crop | `img[y1:y2, x1:x2]` | Focus on a region (ROI) |
| Rotate | `cv2.rotate`, `cv2.warpAffine` | Fix orientation, augmentation |
| Flip | `cv2.flip` | Augmentation |
| Normalize | `/ 255.0`, `(x - mean) / std` | Stable and faster training |
| Color conversion | `cv2.cvtColor` | Grayscale, HSV segmentation, etc. |
| Display | `plt.imshow` | Always convert BGR → RGB |
| Histogram | `cv2.calcHist`, `plt.hist` | Brightness and contrast analysis |
| Pixel access | `img[row, col]` | Debugging and understanding the data |

## Key Takeaways

1. An image is a NumPy array: `(height, width, channels)`, values `0-255`.
2. OpenCV uses **BGR**; Matplotlib uses **RGB**. Always convert before displaying.
3. `cv2.resize` takes `(width, height)`, but array indexing uses `[row, col]`.
4. Convert to `float32` **before** normalizing.
5. Always visualize after each operation to verify the result.
