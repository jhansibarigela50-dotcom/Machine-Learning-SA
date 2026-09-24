# IADAI2011000480-Jhansi Barigela
# ParkVision AI- Intelligent Urban Parking Analytics & Space Optimisation Platform
# Candidate Name: Jhansi Barigela
# Candidate Registration Number - 1000480
# CRS Name: Artificial Intelligence
# Course Name - Machine Learning and Deep Learning
# School name - Birla Open Minds International School, Kollur
# Summative Assessment
# Project Overview
**ParkVision AI** is an end-to-end computer vision and deep learning platform built for smart cities to solve real-time urban parking challenges. Developed for **UrbanFlow AI Pvt. Ltd.**, this platform automates parking slot occupancy detection, calculates real-time space metrics, assesses traffic/congestion levels, and provides actionable recommendations to motorists to reduce unnecessary congestion, fuel waste, and time spent searching for parking.

# Problem Statement
In modern urban environments, the lack of real-time parking availability information creates severe traffic congestion, increases fuel consumption, and leads to driver frustration. Drivers spend significant time searching for open parking spaces without prior knowledge of lot occupancy.   To solve this, there is a critical need for an automated, visual analytics solution capable of evaluating parking lot camera feeds at the slot level. The system must accurately distinguish between empty and occupied parking spaces under varying weather and lighting conditions, automatically compute occupancy metrics and congestion levels, and present real-time recommendations through an accessible, user-friendly interface to optimize urban parking management.

# Project Objectives
Implement Slot-Level Occupancy Detection: Develop and train a deep learning/computer vision model (such as YOLO or MobileNet) capable of accurately classifying individual parking slots as either occupied or empty across diverse environmental conditions.   

Provide Real-Time Availability Metrics: Automatically calculate and display essential parking lot metrics, including total capacity, total occupied slots, and current available spaces from input images.

Generate Visual Overlay Annotations: Integrate automated image processing to highlight parking spaces with clear color-coded bounding boxes (green for empty, red for occupied) for instant visual interpretation.   

Compute Utilization Analytics & Congestion Levels: Implement logic to calculate occupancy percentages and categorize lot status into low, moderate, or high congestion tiers.   

Deliver Intelligent Decision Recommendations: Build an automated decision-making layer that provides actionable feedback to drivers based on current occupancy (e.g., suggesting whether to enter or seek alternative parking).   

Build and Deploy an Interactive Dashboard: Design an intuitive Streamlit web application that accepts image uploads, displays detection outputs and metrics, and deploy it to Streamlit Cloud for public access. 

# Data Preparation
1. Dataset Selection & Balancing
Source & Scope: A subset of high-resolution parking lot images captured under varying environmental conditions (sunny, cloudy, and rainy weather) was curated to ensure the model generalizes across diverse lighting and atmospheric conditions.   

Class Balance: The dataset was structured to contain an even distribution across two classes: occupied and empty. A minimum of 100+ representative images per class was selected to avoid class imbalance and prevent model bias toward any single state.   


2. Image Preprocessing & Standardization
To ensure the deep learning model receives uniform input data, all raw images underwent the following transformation steps:

Spatial Resizing: Every slot image was resized to a standardized input dimension of 224×224 pixels. This matches the expected input vector format for deep neural network architectures (e.g., MobileNet/EfficientNet) and reduces computational overhead during training.   


Pixel Normalization: Pixel intensity values were scaled from the standard [0,255] range down to [0,1] to stabilize gradient descent and accelerate model convergence during training.   


3. Data Augmentation
To make the model resilient against dynamic real-world parking conditions—such as vehicle shadows, camera angle shifts, and lighting variations—dynamic spatial and color augmentations were applied to the training set:   


Spatial Transformations: Random horizontal flips and small degrees of rotation (±15 
∘
 ) were applied to simulate slight perspective shifts.   


Illumination Shifts: Random contrast and brightness adjustments were introduced to mimic changing sunlight, cloud cover, and shadow angles across different times of day.   


4. Dataset Splitting & Folder Hierarchy
The curated dataset was split into three distinct, non-overlapping subsets to ensure reliable training, tuning, and evaluation:   


Training Set (70%): Used by the model to learn visual features and parameters.   


Validation Set (15%): Used during training to monitor loss, prevent overfitting, and tune hyperparameters.   


Testing Set (15%): Kept entirely unseen during development to evaluate final model performance on real-world test cases.   


The files were organized into standard classification directory structures to ensure seamless compatibility with machine learning pipelines:   


Plaintext
dataset/
├── train/
│   ├── occupied/
│   └── empty/
├── val/
│   ├── occupied/
│   └── empty/
└── test/
    ├── occupied/
    └── empty/
