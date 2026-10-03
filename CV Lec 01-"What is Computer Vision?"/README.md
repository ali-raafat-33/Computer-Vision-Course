# Introduction to Computer Vision

A concise, practical introduction to **Computer Vision (CV)**: what it is, how it relates to AI, Machine Learning and Deep Learning, where it is used in the real world, and the four core tasks every practitioner should know.

## Table of Contents

1. [What is Computer Vision?](#1-what-is-computer-vision)
2. [AI vs Machine Learning vs Deep Learning vs Computer Vision](#2-ai-vs-machine-learning-vs-deep-learning-vs-computer-vision)
3. [Real-World Applications](#3-real-world-applications)
4. [Core Computer Vision Tasks](#4-core-computer-vision-tasks)
   - [Image Classification](#41-image-classification)
   - [Object Detection](#42-object-detection)
   - [Image Segmentation](#43-image-segmentation)
   - [Image Generation](#44-image-generation)
5. [Task Comparison](#5-task-comparison)
6. [Further Learning](#6-further-learning)

---

## 1. What is Computer Vision?

**Computer Vision** is a field of Artificial Intelligence that enables computers to **extract meaning from images and videos** and act on it, much like human vision does.

To a computer, an image is just a grid of numbers. A grayscale image is a 2D matrix of pixel intensities (0–255), and a color image is three such matrices (Red, Green, Blue). Computer Vision turns these raw numbers into answers such as *"this is a cat"*, *"there are 3 cars at these locations"*, or *"this region is a tumor"*.

### A typical pipeline

![Computer Vision pipeline](images/cv-pipeline.svg)

| Stage | Purpose |
|---|---|
| **Input image** | Photo, video frame, scan, or camera stream |
| **Pre-processing** | Resize, normalize, augment, and clean the data |
| **Feature extraction** | Learn visual patterns (edges → textures → shapes → objects) using a CNN or Vision Transformer |
| **Task head** | A final layer specialized for the task (classification, detection, segmentation) |
| **Prediction** | Labels, bounding boxes, masks, or a generated image |

---

## 2. AI vs Machine Learning vs Deep Learning vs Computer Vision

These terms are often used interchangeably, but they describe different things.

![AI vs ML vs DL vs CV](images/ai-ml-dl-cv.svg)

| Term | Definition | Example |
|---|---|---|
| **Artificial Intelligence (AI)** | The broad goal of building machines that perform tasks requiring human-like intelligence | Chess engines, chatbots, recommendation systems |
| **Machine Learning (ML)** | A subset of AI where systems **learn patterns from data** instead of being explicitly programmed | Spam filter, house-price prediction |
| **Deep Learning (DL)** | A subset of ML using **multi-layer neural networks** to learn complex representations automatically | CNNs, Transformers, GANs, Diffusion models |
| **Computer Vision (CV)** | An **application domain** of AI focused on understanding visual data | Face unlock, medical imaging, self-driving cars |

**Key takeaways**

- **AI ⊃ ML ⊃ DL**: each one is a subset of the previous one.
- **Computer Vision is a field, not a level.** It is defined by its *input* (images/video), not by its technique.
- Modern CV is dominated by Deep Learning, but classical methods (edge detection, SIFT, HOG, thresholding) are still used where data or compute is limited.

---

## 3. Real-World Applications

![Computer Vision applications](images/applications.svg)

| Industry | Use cases |
|---|---|
| **Healthcare** | Detecting diseases in X-rays/MRI/CT scans, segmenting tumors, counting cells in microscopy |
| **Autonomous vehicles** | Detecting pedestrians, vehicles and traffic signs; lane and drivable-area segmentation |
| **Retail & e-commerce** | Visual search ("find similar products"), shelf monitoring, cashier-less stores |
| **Security** | Face recognition, intrusion detection, license plate recognition |
| **Agriculture** | Crop disease detection, yield estimation, drone-based field mapping |
| **Manufacturing** | Automated defect inspection, assembly verification, robotic guidance |

---

## 4. Core Computer Vision Tasks

![Core CV tasks](images/cv-tasks.svg)

The four tasks differ in **how detailed the output is**: from a single label per image, to boxes, to per-pixel masks, to entirely new images.

### 4.1 Image Classification

**Question answered:** *What is in this image?*

The model receives an image and assigns it **one label** (or a probability for each possible label) from a predefined set of classes.

- **Input:** an image
- **Output:** a class label and confidence, e.g. `Cat: 97%`
- **Typical models:** ResNet, EfficientNet, MobileNet, Vision Transformer (ViT)
- **Metrics:** Accuracy, Precision, Recall, F1-score, Top-5 accuracy
- **Examples:** Cat vs dog, pneumonia vs normal X-ray, ripe vs unripe fruit

```python
# Minimal example with a pretrained model (PyTorch / torchvision)
import torch
from torchvision import models, transforms
from PIL import Image

weights = models.ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights).eval()
preprocess = weights.transforms()

img = preprocess(Image.open("cat.jpg")).unsqueeze(0)
with torch.no_grad():
    probs = model(img).softmax(dim=1)

top = probs.argmax().item()
print(weights.meta["categories"][top], probs[0, top].item())
```

> **Limitation:** classification assumes one main subject and does not say *where* the object is.

### 4.2 Object Detection

**Question answered:** *What objects are present, and where are they?*

Detection finds **multiple objects** in an image and draws a **bounding box** around each one with a class label and confidence score.

- **Input:** an image
- **Output:** a list of `(class, confidence, x, y, width, height)`
- **Typical models:** YOLO family, Faster R-CNN, SSD, RT-DETR
- **Metrics:** IoU (Intersection over Union), mAP (mean Average Precision)
- **Examples:** Counting cars in traffic, detecting people on a construction site, locating products on a shelf

```python
# Minimal example with Ultralytics YOLO
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
results = model("street.jpg")
results[0].show()
```

> **Key concept, IoU:** the overlap between the predicted box and the ground-truth box divided by their union. A prediction is usually counted as correct when IoU ≥ 0.5.

### 4.3 Image Segmentation

**Question answered:** *Which pixels belong to which object or class?*

Segmentation gives the **most detailed** understanding by classifying **every pixel**. It comes in three flavors:

| Type | Description | Example |
|---|---|---|
| **Semantic segmentation** | Every pixel gets a class; all objects of the same class share one label | All "road" pixels are one region |
| **Instance segmentation** | Separates individual objects of the same class | "Car 1", "Car 2", "Car 3" each get their own mask |
| **Panoptic segmentation** | Combines both: instances for objects, semantic labels for background | Individual people + sky + road |

- **Typical models:** U-Net, DeepLab, Mask R-CNN, SAM (Segment Anything Model)
- **Metrics:** IoU / Jaccard index, Dice coefficient, pixel accuracy
- **Examples:** Tumor boundary outlining, background removal, drivable-area detection, satellite land-cover mapping

### 4.4 Image Generation

**Question answered:** *Can we create a new, realistic image?*

Unlike the other three tasks, which **analyze** images, generation **creates** them. The model learns the distribution of real images and samples new ones from it, optionally guided by a text prompt or another image.

| Approach | Idea |
|---|---|
| **GANs** (Generative Adversarial Networks) | A generator and a discriminator compete, so the generator learns to produce realistic images |
| **VAEs** (Variational Autoencoders) | Encode images into a compact latent space, then decode samples from it |
| **Diffusion models** | Start from noise and gradually denoise it into an image (Stable Diffusion, DALL·E, Midjourney) |

- **Input:** noise, a text prompt, or a source image
- **Output:** a new image
- **Metrics:** FID (Fréchet Inception Distance), CLIP score, human evaluation
- **Examples:** Text-to-image, image editing and inpainting, super-resolution, synthetic training data

> **Responsible use:** generative models can produce misleading content (deepfakes) and may reflect biases in their training data. Use them with clear disclosure and appropriate safeguards.

---

## 5. Task Comparison

| | Classification | Object Detection | Segmentation | Generation |
|---|---|---|---|---|
| **Goal** | What is it? | What and where? | Which pixel is what? | Create a new image |
| **Output** | 1 label | Boxes + labels | Pixel masks | New image |
| **Detail level** | Low | Medium | High | n/a (synthesis) |
| **Annotation cost** | Low | Medium | High | Usually large unlabeled sets |
| **Example model** | ResNet, ViT | YOLO, Faster R-CNN | U-Net, Mask R-CNN, SAM | Stable Diffusion, GANs |
| **Main metric** | Accuracy / F1 | mAP | IoU / Dice | FID |

**Choosing the right task:** use *classification* when you only need a label, *detection* when you need object locations or counts, *segmentation* when exact shape or area matters, and *generation* when you need to synthesize new visual content.

---

## 6. Further Learning

- [Stanford CS231n: Deep Learning for Computer Vision](http://cs231n.stanford.edu/)
- [PyTorch Vision documentation](https://pytorch.org/vision/stable/index.html)
- [Ultralytics YOLO documentation](https://docs.ultralytics.com/)
- [Hugging Face Diffusers](https://huggingface.co/docs/diffusers)
- [OpenCV documentation](https://docs.opencv.org/)

---
