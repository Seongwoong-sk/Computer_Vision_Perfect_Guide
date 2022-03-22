# Computer_Vision_Perfect_Guide

# 딥러닝 컴퓨터 비전 완벽 가이드

**Inflearn**에서 제공하는 [[개정판] 딥러닝 컴퓨터 비전 완벽 가이드](https://www.inflearn.com/course/%EB%94%A5%EB%9F%AC%EB%8B%9D-%EC%BB%B4%ED%93%A8%ED%84%B0%EB%B9%84%EC%A0%84-%EC%99%84%EB%B2%BD%EA%B0%80%EC%9D%B4%EB%93%9C/dashboard)를 공부하고 정리한 문서입니다.    

**Framework**는 **Tensorflow 2.x**와 **Pytorch 1.10**으로 진행되었습니다.

      
## 목차


### Section I. Understanding of Object Detection

- [x] Object Detection과 Segmentation
- [x] Main factors of Object detection and reason why Object detection is difficult
- [x] Object Localization and Detection 
- [x] Understanding of Regison Proposal and Comparison with Sliding Window
- [x] Region Proposal - Selective Search
- [x] [Understanding of IOU(Intersection over Union) and Coding](https://github.com/Seongwoong-sk/Computer_Vision_Perfect_Guide/blob/main/I.%20Preliminary/selective_search_n_iou.ipynb)
- [x] Understanding of NMS(Non Max Supression)
- [x] Metrics of Object Detection - Precision, Recall, AP(Average Precision), mAP(Mean Average Precision)




### Section II. Major Datasets for Object Detection and Segmentation & OpenCV

- [x] Major Datasets of Object Detection and Pascal VOC Dataset
- [x] [Exploration into Pascal VOC Dataset](https://github.com/Seongwoong-sk/Computer_Vision_Perfect_Guide/blob/main/I.%20Preliminary/pascal_voc_dataset_annotation.ipynb) 
- [x] MS-COCO Dataset
- [x] Introduction to OpenCV
- [x] [Image and Video Practice utilizing OpenCV](https://github.com/Seongwoong-sk/Computer_Vision_Perfect_Guide/blob/main/I.%20Preliminary/opencv_image_n_video.ipynb)
- [x] Introduction to Networks of Object Detection and FPS, Resolution and Trade-off



### Section III. Object Detector affiliated with RCNN (RCNN,SPPNet,Fast RCNN, Faster RCNN)

- [x] RCNN -> Object Detection Models based on Region Proposal
- [x] RCNN -> RCNN Training and Loss
- [x] SPPNet -> Problems of RCNN and Spatial Pyramid Matching
- [x] SPPNet -> Object Detection utilizaing Spatial Pyramid Pooling
- [x] Fast RCNN -> Anchor Box, Region Proposal Network Architecture with Anchor Box
- [x] Fast RCNN -> Training and Performance Comparison
- [x] [Image and VIdeo Practice of OpenCV Deep Neural Network](https://github.com/Seongwoong-sk/Computer_Vision_Perfect_Guide/blob/main/II.%20Faster_RCNN/opencv_faster_rcnn_inference.ipynb)
- [x] Architecture of Modern Object Detection Models

### Section IV. Understanding of MMDetection and Application Practice of Faster RCNN 01

- [x] Packages based on Pytorch of Object Detection/Segmentation
- [x] Introduction to MMDetection
- [x] [MMDetection Image and Video Inference through Faster-RCNN Pretrained Model](https://github.com/Seongwoong-sk/Computer_Vision_Perfect_Guide/blob/main/II.%20Faster_RCNN/mm_faster_rcnn_inference.ipynb)
- [x] [MMDetection Training through Tiny Kitti Dataset](https://github.com/Seongwoong-sk/Computer_Vision_Perfect_Guide/blob/main/II.%20Faster_RCNN/mm_faster_rcnn_train_kitti.ipynb) 
- [x] --> Understanding of MMDetection Dataset, kitti data format and Config 
- [x] --> CustomDataset Mechanism, Making CustomDataset, Setting Config, Image and Video Inference



### Section V. Understanding of MMDetection and Application Practice of Faster RCNN 02

- [x] Understanding of MMDetection Config 
- [x] --> Hierarchical Classification and Major Setting
- [x] --> Data Pipeline and 
- [x] [MMDetection Training through Oxford Pet Dataset](https://github.com/Seongwoong-sk/Computer_Vision_Perfect_Guide/blob/main/II.%20Faster_RCNN/mm_faster_rcnn_train_oxford_pet.ipynb)
- [x] --> Explanation of the Dataset, Making Meta File for MMDetection, Making CustomDataset
- [x] --> Setting Config, Training, Checking the result and Inference
- [x] [MMDetection Training through BCCD(Blood Cell Count and Detection) Dataset](https://github.com/Seongwoong-sk/Computer_Vision_Perfect_Guide/blob/main/II.%20Faster_RCNN/mm_faster_rcnn_train_coco_bccd.ipynb)
- [x] --> Conversion of VOC Format Dataset into COCO Format Dataset
- [x] --> Conversion of Dataset, Config Setting, Training, Inference, Test Data Inference and Evaluation



### Section VI. Concept of SSD(Single Shot Detector) 

- [x] Understanding of SSD
- [x] --> One Stage Detector, SSD and Multi Scale Feature Map
- [x] --> Network Architecture of SSD and Utilization of Multi Scale Feature Map/Anchor Box


### Section VII. SSD Practice

- [x] [SSD Inference through OpenCV](https://github.com/Seongwoong-sk/Computer_Vision_Perfect_Guide/blob/main/III.%20SSD(Single%20Shot%20Detector)/opencv_ssd_inference.ipynb)
- [x] Outline of Tensorflow Hub
- [x] [SSD Inference through Pretrained Model of TF Hub](https://github.com/Seongwoong-sk/Computer_Vision_Perfect_Guide/blob/main/III.%20SSD(Single%20Shot%20Detector)/tf_hub_ssd_inference.ipynb)


### Section VIII. YOLO(You Only Look Once)

- [x] Outline of YOLO
- [x] Understanding of YOLO v1
- [x] Understanding of YOLO v2
- [x] Understanding of YOLO v3
- [x] Implementation Summary of YOLO Object Detection through OpenCV DNN
- [x] [YOLO Inference with OpenCV DNN](https://github.com/Seongwoong-sk/Computer_Vision_Perfect_Guide/blob/main/IV.%20YOLO(You%20Only%20Look%20Once)/opencv_yolov3_inference.ipynb)



### Section IX. Ultralytics YOLO Practice 01

- [x] Outline of Ultralytics YOLO v3
- [x] [YOLO v3 Image and Video Inference](https://github.com/Seongwoong-sk/Computer_Vision_Perfect_Guide/blob/main/IV.%20YOLO(You%20Only%20Look%20Once)/yolov3_inference.ipynb)
- [x] [Ultralytics YOLO v3 Training with coco128 Dataset](https://github.com/Seongwoong-sk/Computer_Vision_Perfect_Guide/blob/main/IV.%20YOLO(You%20Only%20Look%20Once)/yolov3_train_coco.ipynb) 
- [x] --> train.py, Application of wandb(weight and bias), Summary of Train Process of Ultralytics Yolo Package
- [x] --> Understanding Difference between Relative Path and Absolute Path of Config & Weight file
- [x] --> Structure of Data Directory and Mapping Data Config
- [x] [Ultralytics YOLO v3 Training with Oxford Pet Dataset](https://github.com/Seongwoong-sk/Computer_Vision_Perfect_Guide/blob/main/IV.%20YOLO(You%20Only%20Look%20Once)/yolov3_train_oxpet.ipynb)
- [x] --> Exploration of the Dataset and Making Data Structure of Train and Valid Data
- [x] --> Making Meta Data For Train and Valid Data
- [x] --> Conversion of Annotation into Ultralytics YOLO Format
- [x] --> Application of Dataset yaml, Training and Result Review
- [x] --> Image and Video Inference and Evaluation of Test Data


### Section X. Ultralytics YOLO Practice 02

- [x] Making Custom Dataset by Annotation tool - CVAT(Computer Vision Annotation Tool)
- [x] [Ultralytics YOLO v3 Training and Inference Incredibles Dataset made by CVAT](https://github.com/Seongwoong-sk/Computer_Vision_Perfect_Guide/blob/main/IV.%20YOLO(You%20Only%20Look%20Once)/yolov3_train_incredibles.ipynb)
- [x] [YOLO v5 Training Pracetice with BCCD Dataset](https://github.com/Seongwoong-sk/Computer_Vision_Perfect_Guide/blob/main/IV.%20YOLO(You%20Only%20Look%20Once)/yolov5_train_bccd.ipynb)
- [x] --> Outline of Coco2Yolo Dataset Conversion
- [x] --> Conversion of COCO Dataset into Ultralytics YOLO Format
- [x] --> Application of Dataset yaml, Train and Inference


### Section XI. RetinaNet and EfficientDet

- [x] Outline of RetinaNet
- [x] Understanding of RetinaNet - Focal Loss and Feature Pyramid Network
- [x] Outline of EfficientDet
- [x] Understanding of EfficientDet - BiFPN, EfficientNet, Compound Scaling and Performance Evaluation


### Section XII. AutoML EfficientDet Practice 01

- [x] Introduction to AutoML EfficientDet Package
- [x] [Inference Practice utilizing EfficientDet and EfficientDet Lite Model of Tensorflow Hub](https://github.com/Seongwoong-sk/Computer_Vision_Perfect_Guide/blob/main/V.%20Efficientdet/tf_hub_efficientdet_inference.ipynb)
- [x] [Inference utilizing AutoML EfficientDet](https://github.com/Seongwoong-sk/Computer_Vision_Perfect_Guide/blob/main/V.%20Efficientdet/efficientdet_inference.ipynb) 
- [x] --> Understanding of Structure of AutoML Package and Install
- [x] --> Result Review, Visulalization and Resize images


### Section XIII. AutoML EfficientDet Practice 02

- [x] [Pascal VOC Training Practice utilizing AutoML EfficientDet](https://github.com/Seongwoong-sk/Computer_Vision_Perfect_Guide/blob/main/V.%20Efficientdet/efficientdet_train_pascal_voc.ipynb)
- [x] --> Making TFRecords Dataset, Setting Config for Training
- [x] --> Making Model and Dataset, Training, Result Review and Inference
- [x] [Esri Object Detection Challenge Practice utilizing AutoML EfficientDet](https://github.com/Seongwoong-sk/Computer_Vision_Perfect_Guide/blob/main/V.%20Efficientdet/efficientdet_train_esri_challenge.ipynb)
- [x] --> Understanding of TFRecrod, looking at Esri Dataset 
- [x] --> Conversion of Esri Dataset into TFRecord
- [x] --> Setting Config, Train, Result Review and Inference


### Section XIV. Segmentation - Mask RCNN

- [x] Outline of Segmentation
- [x] Understanding of Semantic Segmentation FCN
- [x] Understanding of Mask RCNN


### Section XV. Mask RCNN Practice 01

- [x] Deep Understanding of MS COCO Dataset
- [x] Understanding and Utilization of Pycocotools
- [x] [Visualization through Pycocotools and of Polygon and Mask Segmentation](https://github.com/Seongwoong-sk/Computer_Vision_Perfect_Guide/blob/main/VI.%20Mask_RCNN/coco_annotations_mask_visuals.ipynb)
- [x] [Mask RCNN Inference utilizing Opencv DNN](https://github.com/Seongwoong-sk/Computer_Vision_Perfect_Guide/blob/main/VI.%20Mask_RCNN/opencv_mask_rcnn_inference.ipynb)
- [x] --> Result Reivew, Visualization of Mask Results, Functionalization of Inference Visualization Code and Video Inference
- [x] [Mask RCNN Inference utilizing MMDetection](https://github.com/Seongwoong-sk/Computer_Vision_Perfect_Guide/blob/main/VI.%20Mask_RCNN/mm_mask_rcnn_inference.ipynb)
- [x] --> Config Setting for Mask RCNN, Inference, Result Review, Video Inference and Visualization of Inference Results as Segmentation



### Section XVI. Mask RCNN Practice 02

- [x] [Mask RCNN Training Practice with Pascal VOC Dataset](https://github.com/Seongwoong-sk/Computer_Vision_Perfect_Guide/blob/main/VI.%20Mask_RCNN/mm_mask_rcnn_train_pascal_voc.ipynb)
- [x] --> Conversion of Pascal VOC Dataset into COCO Data Format
- [x] --> Setting Dataset and Config for Train and Train
- [x] --> Image and Video Inference with Trained Model
- [x] --> Logic of Conversion of Pascal VOC into COCO Data Format -> Dataset-Converter  Utility
- [x] [Mask RCNN Training Practice with Ballon Dataset](https://github.com/Seongwoong-sk/Computer_Vision_Perfect_Guide/blob/main/VI.%20Mask_RCNN/mm_mask_rcnn_train_balloon.ipynb)
- [x] --> Understanding of the Dataset, Conversion of Ballon Annotation to COCO Format Annotation
- [x] --> Train and Segmentation with Gray Scale Filter
- [x] [Kaggle Nucleus Segmentation Challenge](https://github.com/Seongwoong-sk/Computer_Vision_Perfect_Guide/blob/main/VI.%20Mask_RCNN/mm_mask_rcnn_train_nucleus.ipynb)
- [x] --> Looking at Nucleus Dataset, Coordinate Extraction of Segmentation Polygon 
- [x] --> Conversion of Anntation into COCO Data Format, Segmentation Visualization of COCO JSON through Pycocotools
- [x] --> Config Setting, Training, Result Review and Inference
