# Module 1 — Computer Vision Foundations

> **Goal of this module:** understand what Computer Vision is, how a computer "sees" an image (spoiler: as numbers), and how to load, inspect, transform, and save images using **OpenCV** and **Pillow**.

By the end you will be able to:

- Explain what Computer Vision is and where it is used
- Describe an image as an array of numbers (a tensor)
- Choose the right image format for the job
- Resize, crop, rotate, flip, normalize, and convert color spaces
- Visualize images, histograms, and individual pixel values
- Build a small image-processing tool of your own

---

## Table of Contents

1. [What is Computer Vision?](#11-what-is-computer-vision)
2. [How Computers See Images](#12-how-computers-see-images)
3. [Image Representation](#13-image-representation)
4. [Image Formats](#14-image-formats)
5. [Image Operations](#15-image-operations)
6. [Image Visualization](#16-image-visualization)
7. [Introduction to OpenCV](#17-introduction-to-opencv)
8. [PIL / Pillow](#18-pil--pillow)
9. [Mini Project: Image Processing Playground](#mini-project--image-processing-playground)
10. [Cheat Sheet](#cheat-sheet)
11. [Common Mistakes](#common-mistakes)
12. [Practice Questions](#practice-questions)

---

## Setup

```bash
pip install opencv-python pillow numpy matplotlib
```

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
```

---

## 1.1 What is Computer Vision?

**Computer Vision (CV)** is the field of AI that teaches computers to **understand images and videos**: to recognize what is in them, where things are, and what is happening.

Humans do this effortlessly. For a computer, an image is just a grid of numbers, so we need algorithms that turn those numbers into meaning.

### AI vs ML vs DL vs CV

These terms are often mixed up. Think of them as nested circles, plus Computer Vision as an *application area* that cuts across them.

```
┌──────────────────────────────────────────────┐
│  Artificial Intelligence (AI)                │
│  Machines that perform "intelligent" tasks   │
│  ┌────────────────────────────────────────┐  │
│  │  Machine Learning (ML)                 │  │
│  │  Learn patterns from data              │  │
│  │  ┌──────────────────────────────────┐  │  │
│  │  │  Deep Learning (DL)              │  │  │
│  │  │  Neural networks, many layers    │  │  │
│  │  └──────────────────────────────────┘  │  │
│  └────────────────────────────────────────┘  │
└──────────────────────────────────────────────┘

Computer Vision = AI applied to images/video
(uses classical methods AND ML/DL)
```

| Term | Meaning | Example |
|------|---------|---------|
| **AI** | Any technique that makes machines act intelligently | A chess engine |
| **ML** | Systems that learn from data instead of fixed rules | Spam filter |
| **DL** | ML using deep neural networks | Face recognition |
| **CV** | Extracting understanding from images/video | Reading a license plate |

> **Key point:** Computer Vision is not a *subset* of deep learning. Many CV tools (like the ones in this module) are classical image processing and need no learning at all. Deep learning is simply the most powerful tool CV uses today.

### Real-World Applications

| Domain | Application |
|--------|-------------|
| Healthcare | Detecting tumors in X-rays and MRI scans |
| Automotive | Self-driving cars detecting pedestrians and lanes |
| Security | Face unlock, surveillance, access control |
| Retail | Cashier-less stores, shelf monitoring |
| Agriculture | Detecting plant diseases from leaf photos |
| Banking / Government | Extracting data from ID cards and documents (OCR) |
| Manufacturing | Spotting defects on a production line |
| Social media | Photo filters, auto-tagging, content moderation |

### The Four Core Tasks

| Task | Question it answers | Output |
|------|---------------------|--------|
| **Image Classification** | *What is in this image?* | One label ("cat") |
| **Object Detection** | *What is where?* | Boxes + labels |
| **Image Segmentation** | *Which pixels belong to what?* | A label for every pixel |
| **Image Generation** | *Can you create a new image?* | A brand-new image |

```
Classification      Detection            Segmentation         Generation
┌───────────┐      ┌───────────┐        ┌───────────┐        "a cat wearing
│           │      │ ┌─────┐   │        │ ░░░░░░░   │         a hat" ──► 🖼️
│    🐱     │      │ │ 🐱  │cat│        │ ░░🐱░░░   │
│           │      │ └─────┘   │        │ ░░░░░░░   │
└───────────┘      └───────────┘        └───────────┘
   "cat"           box + "cat"          every pixel labeled
```

---

## 1.2 How Computers See Images

### Digital Images

A digital image is a **grid of tiny squares called pixels**. Each pixel stores a number (or a few numbers) describing its brightness or color. Zoom in far enough on any photo and you will see the squares.

### Pixels

**Pixel** = *Picture Element*, the smallest unit of an image.

### Image Dimensions: Height × Width × Channels

Every image has three dimensions:

- **Height**: number of rows of pixels
- **Width**: number of columns of pixels
- **Channels**: number of values stored per pixel

An image with shape `(480, 640, 3)` is 480 pixels tall, 640 pixels wide, with 3 channels.

> ⚠️ **Order matters:** in NumPy / OpenCV the shape is **(Height, Width, Channels)**, *not* (Width, Height). This is the most common source of confusion.

```python
img = cv2.imread("cat.jpg")
print(img.shape)   # (480, 640, 3) → H, W, C
h, w, c = img.shape
```

### Grayscale Images

- **1 channel**, one number per pixel
- Represents brightness: `0` = black, `255` = white, values in between = shades of gray
- Shape: `(H, W)` (or `(H, W, 1)`)

```
Grayscale 3×3 example:

  0   128  255          black   gray   white
 64   192   32
255     0  128
```

### RGB Images

- **3 channels**: Red, Green, Blue
- Every color is a mix of these three primary lights
- Each pixel is like `[R, G, B]`

| Color | R | G | B |
|-------|---|---|---|
| Red | 255 | 0 | 0 |
| Green | 0 | 255 | 0 |
| Blue | 0 | 0 | 255 |
| White | 255 | 255 | 255 |
| Black | 0 | 0 | 0 |
| Yellow | 255 | 255 | 0 |

### Pixel Values and 8-bit Images

In a standard **8-bit image**, each value is stored in 8 bits, so it can be one of 2⁸ = **256 values**: from `0` to `255`.

- `0` → no intensity
- `255` → full intensity
- Data type in NumPy: `uint8` (unsigned 8-bit integer)

```python
print(img.dtype)       # uint8
print(img.min(), img.max())   # e.g. 0 255
```

### Image Tensors

A **tensor** is just a multi-dimensional array of numbers. An RGB image *is* a 3D tensor:

```
shape = (Height, Width, Channels)

            ┌──────────── Width ───────────┐
          ┌ ┌───┬───┬───┬───┬───┐
  Height  │ │   │   │   │   │   │   each cell = [R, G, B]
          │ ├───┼───┼───┼───┼───┤
          │ │   │   │   │   │   │
          └ └───┴───┴───┴───┴───┘
```

```python
print(type(img))   # <class 'numpy.ndarray'>
print(img.ndim)    # 3
print(img.size)    # H × W × C total numbers
```

> **Teaching tip for students:** ask them *"How many numbers are in a 1920×1080 RGB photo?"* → 1920 × 1080 × 3 = **6,220,800 numbers!**

---

## 1.3 Image Representation

### Channels

Channels are the "layers" stacked on top of each other. For RGB, you have a red layer, a green layer, and a blue layer.

```python
b, g, r = cv2.split(img)   # OpenCV order is B, G, R (see warning below)
print(r.shape)             # (H, W): a single channel is a 2D grayscale-like image
```

### Channel Ordering: RGB vs BGR

| Library | Default channel order |
|---------|----------------------|
| **OpenCV** (`cv2`) | **BGR** |
| **Pillow**, Matplotlib, PyTorch, TensorFlow | **RGB** |

> 🚨 **The #1 beginner bug:** load with OpenCV, show with Matplotlib → colors look wrong (blue skin, orange sky). Fix it by converting:
>
> ```python
> img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
> ```

### HWC vs CHW

Different libraries expect the dimensions in a different order:

| Format | Order | Used by |
|--------|-------|---------|
| **HWC** | Height, Width, Channels | OpenCV, NumPy, Matplotlib, Pillow, TensorFlow (default) |
| **CHW** | Channels, Height, Width | **PyTorch** |

```python
# HWC → CHW  (what PyTorch expects)
chw = np.transpose(img_rgb, (2, 0, 1))
print(img_rgb.shape)  # (480, 640, 3)
print(chw.shape)      # (3, 480, 640)

# CHW → HWC  (to display again)
hwc = np.transpose(chw, (1, 2, 0))
```

### Batch Dimension

Neural networks process **many images at once**, so they add a 4th dimension at the front: the **batch size (N)**.

| Layout | Shape | Framework |
|--------|-------|-----------|
| **NCHW** | (N, C, H, W) | PyTorch |
| **NHWC** | (N, H, W, C) | TensorFlow / Keras |

```python
batch = np.expand_dims(chw, axis=0)
print(batch.shape)   # (1, 3, 480, 640) → 1 image in the batch
```

**Mental model:**

```
Single image :  (H, W, C)       →  (480, 640, 3)
Add batch    :  (N, H, W, C)    →  (1, 480, 640, 3)
PyTorch      :  (N, C, H, W)    →  (1, 3, 480, 640)
```

---

## 1.4 Image Formats

### Comparison

| Format | Compression | Transparency | Best for |
|--------|-------------|--------------|----------|
| **JPG / JPEG** | Lossy | ❌ No | Photos, web, small file size |
| **PNG** | Lossless | ✅ Yes | Screenshots, logos, graphics, text |
| **BMP** | None (usually) | Limited | Simple raw storage, very large files |
| **TIFF** | Lossless (or none) | ✅ Yes | Medical, scientific, print, scanning |

### Lossy vs Lossless Compression

- **Lossy**: makes the file much smaller by **permanently discarding** some information. Each re-save can reduce quality further.
- **Lossless**: makes the file smaller **without losing any information**. You can recover the exact original pixels.

```
Original ──► Lossy (JPEG)     ──► smaller file, tiny details lost forever
Original ──► Lossless (PNG)   ──► bigger file, every pixel preserved
```

### RGB vs RGBA

- **RGB**: 3 channels (Red, Green, Blue)
- **RGBA**: 4 channels, the extra **A = Alpha** controls transparency (`0` = fully transparent, `255` = fully opaque)

```python
img = cv2.imread("logo.png", cv2.IMREAD_UNCHANGED)  # keeps the alpha channel
print(img.shape)   # (H, W, 4) → BGRA in OpenCV
```

> **Why it matters for ML:** most models expect **3 channels**. If you feed an RGBA PNG into a model, you will get a shape error. Always convert (`.convert("RGB")` in Pillow).

### Which Format Should I Use?

- Training data of photos → **JPG** is fine
- Masks for segmentation → **PNG** (lossless! JPEG would corrupt label values)
- Medical / scientific images → **TIFF** or PNG
- Need transparency → **PNG**

---

## 1.5 Image Operations

All examples use OpenCV (`img = cv2.imread("cat.jpg")`).

### Resize

Change the width and height. Needed because neural networks require a **fixed input size** (e.g. 224×224).

```python
resized = cv2.resize(img, (224, 224))   # (width, height) ← note the order!
```

> ⚠️ `cv2.resize` takes **(width, height)**, but `img.shape` gives **(height, width)**. Opposite orders!

Interpolation options: `cv2.INTER_AREA` (best for shrinking), `cv2.INTER_LINEAR` (default), `cv2.INTER_CUBIC` (best for enlarging).

### Crop

Cropping is just **NumPy slicing**: `img[y1:y2, x1:x2]` (rows first, then columns).

```python
cropped = img[100:300, 150:400]   # rows 100-300, columns 150-400
```

### Rotate

```python
# Quick 90° rotations
rot90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

# Any angle
h, w = img.shape[:2]
M = cv2.getRotationMatrix2D((w // 2, h // 2), 45, 1.0)  # center, angle, scale
rotated = cv2.warpAffine(img, M, (w, h))
```

### Flip

```python
flip_h = cv2.flip(img, 1)    # horizontal (mirror left ↔ right)
flip_v = cv2.flip(img, 0)    # vertical   (upside down)
flip_b = cv2.flip(img, -1)   # both
```

Flipping is also a popular **data augmentation** technique: it creates new training samples for free.

### Normalize

Neural networks train better when values are small and consistent. Normalization rescales pixel values.

```python
# 1) Scale to [0, 1]
norm = img.astype(np.float32) / 255.0

# 2) Standardize with mean and std (e.g. ImageNet values, in RGB order)
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
standardized = (norm - mean) / std
```

> Convert to `float32` **before** dividing. Doing math directly on `uint8` can overflow or wrap around.

### Convert Color Spaces

A **color space** is a different way of describing color.

| Color space | Description | Typical use |
|-------------|-------------|-------------|
| **RGB / BGR** | Red, Green, Blue | Display, most DL models |
| **Grayscale** | Brightness only | Simplify, edge detection, OCR |
| **HSV** | Hue, Saturation, Value | Color-based detection (e.g. "find red objects") |
| **LAB** | Perceptual lightness + color | Color correction |

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
```

---

## 1.6 Image Visualization

### Displaying Images (Matplotlib)

```python
img = cv2.imread("cat.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img_rgb)
plt.title("My Image")
plt.axis("off")
plt.show()
```

For grayscale, tell Matplotlib to use a gray colormap, otherwise it applies a colorful default:

```python
plt.imshow(gray, cmap="gray")
```

### Displaying Multiple Images

```python
images = [img_rgb, gray, resized_rgb]
titles = ["Original", "Grayscale", "Resized"]

plt.figure(figsize=(12, 4))
for i, (im, t) in enumerate(zip(images, titles)):
    plt.subplot(1, 3, i + 1)          # 1 row, 3 columns, position i+1
    plt.imshow(im, cmap="gray" if im.ndim == 2 else None)
    plt.title(t)
    plt.axis("off")
plt.tight_layout()
plt.show()
```

### Histograms

A **histogram** shows how many pixels have each intensity value (0-255). It tells you about brightness and contrast at a glance.

- Peak on the **left** → dark image
- Peak on the **right** → bright image
- Narrow spread → low contrast
- Wide spread → good contrast

```python
# Grayscale histogram
plt.hist(gray.ravel(), bins=256, range=(0, 256), color="gray")
plt.title("Grayscale Histogram")
plt.xlabel("Pixel value")
plt.ylabel("Number of pixels")
plt.show()

# Per-channel histogram (RGB)
for i, color in enumerate(("r", "g", "b")):
    hist = cv2.calcHist([img_rgb], [i], None, [256], [0, 256])
    plt.plot(hist, color=color)
plt.title("RGB Histogram")
plt.show()
```

### Pixel Inspection

Look at actual pixel values:

```python
# Pixel at row=100, column=50
print(img_rgb[100, 50])        # e.g. [182  97  45] → [R, G, B]

# A single channel value
print(img_rgb[100, 50, 0])     # Red value

# A small 3×3 patch of grayscale values
print(gray[100:103, 50:53])

# Change a pixel (draw a red dot)
img_rgb[100, 50] = [255, 0, 0]
```

> Remember: indexing is `[row, column]` = `[y, x]`, not `[x, y]`.

---

## 1.7 Introduction to OpenCV

**OpenCV** (Open Source Computer Vision Library) is the most widely used CV library. It is fast (written in C++), and has hundreds of ready-to-use functions.

### Installing

```bash
pip install opencv-python
```

```python
import cv2
print(cv2.__version__)
```

### The Five Essential Functions

| Function | Purpose |
|----------|---------|
| `cv2.imread()` | Read an image from disk |
| `cv2.imshow()` | Display an image in a window |
| `cv2.imwrite()` | Save an image to disk |
| `cv2.resize()` | Resize an image |
| `cv2.cvtColor()` | Convert color space |

### `cv2.imread()`: load an image

```python
img = cv2.imread("cat.jpg")                          # color (BGR)
gray = cv2.imread("cat.jpg", cv2.IMREAD_GRAYSCALE)   # grayscale
full = cv2.imread("logo.png", cv2.IMREAD_UNCHANGED)  # keep alpha channel

if img is None:
    print("Could not load image: check the path!")
```

> ⚠️ `imread` does **not** raise an error if the file is missing. It silently returns `None`. Always check.

### `cv2.imshow()`: display in a window

```python
cv2.imshow("Window title", img)
cv2.waitKey(0)           # wait until a key is pressed
cv2.destroyAllWindows()  # close windows
```

> `cv2.imshow` does not work in Jupyter / Google Colab / servers. Use Matplotlib instead (or `from google.colab.patches import cv2_imshow` in Colab).

### `cv2.imwrite()`: save an image

```python
cv2.imwrite("output.png", img)
cv2.imwrite("output.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])  # control JPEG quality
```

The file extension decides the format.

### `cv2.resize()` and `cv2.cvtColor()`

```python
small = cv2.resize(img, (320, 240))                       # (width, height)
half  = cv2.resize(img, None, fx=0.5, fy=0.5)             # scale by factor
gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

---

## 1.8 PIL / Pillow

**Pillow** is a friendly, Pythonic image library, great for simple image tasks and commonly used for preparing data for deep learning (e.g. in `torchvision`).

```bash
pip install pillow
```

### Opening Images

```python
from PIL import Image

img = Image.open("cat.jpg")
print(img.size)     # (width, height) ← note: opposite of NumPy shape!
print(img.mode)     # 'RGB'
print(img.format)   # 'JPEG'
img.show()          # opens in your default viewer
```

### RGB Conversion

Images can come in other modes (`RGBA`, `L`, `P`, `CMYK`). Convert to a consistent mode:

```python
img = Image.open("logo.png").convert("RGB")   # drops alpha, ensures 3 channels
gray = img.convert("L")                       # grayscale
```

### Resize

```python
resized = img.resize((224, 224))      # (width, height)
```

### Crop

```python
cropped = img.crop((left, upper, right, lower))   # a box of 4 coordinates
cropped = img.crop((100, 50, 400, 300))
```

### Rotate

```python
rotated = img.rotate(45)                   # counter-clockwise, same canvas size
rotated = img.rotate(45, expand=True)      # enlarge canvas so nothing is cut off
flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
```

### Saving Images

```python
img.save("output.png")
img.save("output.jpg", quality=90)
```

### Converting Between Pillow and NumPy

```python
arr = np.array(img)             # PIL → NumPy (H, W, C), RGB order
img2 = Image.fromarray(arr)     # NumPy → PIL
```

### OpenCV vs Pillow

| | OpenCV | Pillow |
|---|--------|--------|
| Channel order | **BGR** | **RGB** |
| Data type | NumPy array | `Image` object |
| Size property | `shape` = (H, W, C) | `size` = (W, H) |
| Speed | Very fast | Good |
| Strengths | Advanced CV (detection, tracking, filters) | Simple loading, converting, saving |
| Resize arg order | (width, height) | (width, height) |

---

## MINI PROJECT — Image Processing Playground

**Task:** Write a script that loads an image, then resizes it, converts it to grayscale, rotates it, crops it, and saves every processed result.

### Requirements

1. Load an image from disk (and handle the "file not found" case)
2. Print its shape, data type, and pixel range
3. Resize it to 300×300
4. Convert it to grayscale
5. Rotate it by 45° (or any angle)
6. Crop a region from the center
7. Save all processed images in an `output/` folder
8. *(Bonus)* Display all results in one Matplotlib figure

### Project Structure

```
image-playground/
├── input/
│   └── cat.jpg
├── output/
├── playground.py
└── README.md
```

### Starter Solution (OpenCV)

```python
import os
import cv2
import matplotlib.pyplot as plt

INPUT_PATH = "input/cat.jpg"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Load
img = cv2.imread(INPUT_PATH)
if img is None:
    raise FileNotFoundError(f"Could not load image at {INPUT_PATH}")

# 2. Inspect
print("Shape :", img.shape)
print("Dtype :", img.dtype)
print("Range :", img.min(), "-", img.max())

# 3. Resize
resized = cv2.resize(img, (300, 300))

# 4. Grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 5. Rotate 45°
h, w = img.shape[:2]
M = cv2.getRotationMatrix2D((w // 2, h // 2), 45, 1.0)
rotated = cv2.warpAffine(img, M, (w, h))

# 6. Crop the center half
y1, y2 = h // 4, 3 * h // 4
x1, x2 = w // 4, 3 * w // 4
cropped = img[y1:y2, x1:x2]

# 7. Save
cv2.imwrite(f"{OUTPUT_DIR}/resized.jpg", resized)
cv2.imwrite(f"{OUTPUT_DIR}/gray.jpg", gray)
cv2.imwrite(f"{OUTPUT_DIR}/rotated.jpg", rotated)
cv2.imwrite(f"{OUTPUT_DIR}/cropped.jpg", cropped)
print("Saved all images to", OUTPUT_DIR)

# 8. Bonus: show everything
results = {
    "Original": cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
    "Resized": cv2.cvtColor(resized, cv2.COLOR_BGR2RGB),
    "Grayscale": gray,
    "Rotated": cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB),
    "Cropped": cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB),
}

plt.figure(figsize=(15, 4))
for i, (title, im) in enumerate(results.items()):
    plt.subplot(1, len(results), i + 1)
    plt.imshow(im, cmap="gray" if im.ndim == 2 else None)
    plt.title(title)
    plt.axis("off")
plt.tight_layout()
plt.show()
```

### Same Idea with Pillow

```python
from PIL import Image
import os

os.makedirs("output", exist_ok=True)

img = Image.open("input/cat.jpg").convert("RGB")
w, h = img.size

img.resize((300, 300)).save("output/resized_pil.jpg")
img.convert("L").save("output/gray_pil.jpg")
img.rotate(45, expand=True).save("output/rotated_pil.jpg")
img.crop((w // 4, h // 4, 3 * w // 4, 3 * h // 4)).save("output/cropped_pil.jpg")
```

### Challenge Ideas

- Add a **flip** (horizontal and vertical)
- Plot the **histogram** before and after grayscale conversion
- Turn the script into a function that processes **every image in a folder**
- Add a command-line argument for the input path (`argparse`)
- Build a small **Gradio** interface with sliders for angle and size

---

## Cheat Sheet

### Shapes and Orders

| Concept | Remember |
|---------|----------|
| NumPy / OpenCV shape | `(Height, Width, Channels)` |
| Pillow `.size` | `(Width, Height)` |
| `cv2.resize` argument | `(Width, Height)` |
| Pixel indexing | `img[row, col]` = `img[y, x]` |
| OpenCV colors | **BGR** |
| Everyone else | **RGB** |
| PyTorch tensor | `(N, C, H, W)` |
| Pixel range (8-bit) | `0 - 255`, dtype `uint8` |

### Quick Code Reference

```python
# OpenCV
img  = cv2.imread(path)
cv2.imwrite(path, img)
img  = cv2.resize(img, (w, h))
img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img  = cv2.flip(img, 1)
crop = img[y1:y2, x1:x2]

# Pillow
img = Image.open(path).convert("RGB")
img = img.resize((w, h))
img = img.crop((l, t, r, b))
img = img.rotate(angle, expand=True)
img.save(path)

# Normalize
x = img.astype(np.float32) / 255.0

# HWC → CHW → add batch
x = np.transpose(x, (2, 0, 1))
x = np.expand_dims(x, 0)
```

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Showing a BGR image with Matplotlib | Weird colors (blue faces) | `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` |
| Wrong file path in `imread` | `img` is `None`, later `AttributeError` | Check `if img is None` |
| Mixing up `(H, W)` and `(W, H)` | Stretched or squashed images | Remember `resize` wants `(W, H)` |
| Normalizing on `uint8` | Overflow, weird values | `astype(np.float32)` first |
| Feeding RGBA images to a model | Shape error (4 channels) | `.convert("RGB")` |
| Saving segmentation masks as JPG | Label values get corrupted | Use PNG |
| Using `cv2.imshow` in Jupyter / Colab | Kernel hangs or crashes | Use `plt.imshow` |
| Forgetting `cmap="gray"` | Grayscale shows in odd colors | `plt.imshow(gray, cmap="gray")` |

---

## Practice Questions

1. What is the difference between AI, Machine Learning, and Deep Learning?
2. Name the four core Computer Vision tasks and give one real example of each.
3. An image has shape `(720, 1280, 3)`. What are its height, width, and number of channels? How many numbers does it contain?
4. What range of values can an 8-bit pixel hold, and why?
5. Why do colors look wrong when you display an OpenCV image with Matplotlib?
6. Convert an image of shape `(224, 224, 3)` to PyTorch format, including the batch dimension. What is the final shape?
7. When would you choose PNG over JPEG? When the reverse?
8. What does the alpha channel in RGBA represent?
9. Write one line of NumPy to crop the top-left 100×100 pixels of an image.
10. What does a histogram that is concentrated on the far left tell you about an image?

---

## What's Next?

In **Module 2** we build on these foundations to go beyond simple operations: filtering, edge detection, and the first steps toward feeding images into neural networks.

> **Remember:** *an image is just a NumPy array.* Everything in this module (resizing, cropping, rotating, normalizing) is simply math on that array.
