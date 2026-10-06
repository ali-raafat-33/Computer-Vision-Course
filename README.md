# Computer Vision Course

A complete path from deep learning to modern computer vision: 11 modules, 9 projects, a final project, and a quiz after every module.

## Simple Roadmap

```
Foundations → Image Processing → CNNs → Transfer Learning → Classification
      → Object Detection → Segmentation → Advanced CV → Modern CV (ViT / VLMs)
            → Generative CV → Deployment → Final Project
```

| Phase | Modules | Goal |
|-------|---------|------|
| 1. Basics | 1–2 | Understand images and process them with OpenCV |
| 2. Deep Learning | 3–5 | Train, fine-tune, and evaluate image classifiers |
| 3. Core Tasks | 6–7 | Detect objects and segment images |
| 4. Advanced | 8–9 | Track objects, read text, use Transformers and VLMs |
| 5. Generative & Production | 10–11 | Generate images and deploy models |

## Modules at a Glance

### Phase 1: Basics

**Module 1: Computer Vision Foundations**
How computers see images (pixels, channels, tensors, HWC vs CHW), image formats, basic operations, OpenCV and Pillow.
**Project:** Image Processing Playground

**Module 2: Image Processing**
Histograms, thresholding, filtering, edge detection (Sobel, Laplacian, Canny), morphology, contours, geometric transformations.
**Project 1:** Document Scanner

### Phase 2: Deep Learning

**Module 3: Deep Learning for CV**
CNN revision, data preparation, augmentation, training loop, evaluation metrics.
**Project 2:** Cats vs Dogs Classifier

**Module 4: Transfer Learning**
Pretrained models (ResNet, EfficientNet, MobileNet...), freezing layers, fine-tuning, choosing a model.
**Project 3:** Transfer Learning Image Classifier

**Module 5: Image Classification**
Binary, multi-class, and multi-label classification, losses, imbalanced data, error analysis, Grad-CAM explainability.
**Project 4:** Real-World Image Classification (data to deployment)

### Phase 3: Core Tasks

**Module 6: Object Detection**
Bounding boxes, IoU, NMS, mAP, two-stage (Faster R-CNN) vs one-stage (YOLO, SSD) detectors.
**Project 5:** YOLO Object Detection

**Module 7: Image Segmentation**
Semantic vs instance segmentation, masks, U-Net, Mask R-CNN, Dice/IoU metrics.
**Project 6:** Medical Image Segmentation

### Phase 4: Advanced

**Module 8: Advanced Computer Vision**
Object tracking (SORT, Deep SORT, ByteTrack), optical flow, pose estimation, face detection, OCR, video processing.
**Project 7:** Real-Time Object Tracking System

**Module 9: Modern Computer Vision**
Vision Transformers (ViT, Swin, ConvNeXt), CLIP, vision-language models (Qwen-VL, LLaVA), captioning, VQA.
**Project 8:** Image Question Answering App

### Phase 5: Generative & Production

**Module 10: Generative Computer Vision**
Autoencoders, VAEs, GANs, diffusion models, Stable Diffusion concepts.
**Project 9:** Image Generation App

**Module 11: Deployment**
Saving models, inference pipelines, FastAPI, Gradio, optimization (quantization, pruning, ONNX, TensorRT), edge devices (mobile, Raspberry Pi, Jetson).
**Final Project:** Complete Computer Vision Application, from problem to final demo

## Projects Summary

| # | Project | Skill Practiced |
|---|---------|-----------------|
| Mini | Image Processing Playground | Basic image operations |
| 1 | Document Scanner | Edges, contours, perspective transform |
| 2 | Cats vs Dogs Classifier | Training a CNN |
| 3 | Transfer Learning Classifier | Fine-tuning pretrained models |
| 4 | Real-World Classification | Full pipeline with Grad-CAM |
| 5 | YOLO Object Detection | Detection and bounding boxes |
| 6 | Medical Image Segmentation | U-Net |
| 7 | Real-Time Object Tracking | YOLO + ByteTrack |
| 8 | Image Question Answering | Vision-language models |
| 9 | Image Generation App | Generative models |
| Final | Complete CV Application | End-to-end, deployed |

## Final Project Pipeline

```
Problem → Dataset → Analysis → Preprocessing → Augmentation → Model Selection
→ Training → Evaluation → Error Analysis → Optimization → Deployment → Demo
```

## Suggested Tools

OpenCV, Pillow, PyTorch, torchvision, YOLO, FastAPI, Gradio, ONNX
