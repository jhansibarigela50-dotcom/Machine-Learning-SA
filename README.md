# IADAI2011000480-Jhansi Barigela
# ParkVision AI- Intelligent Urban Parking Analytics & Space Optimisation Platform
# Candidate Name: Jhansi Barigela
# Candidate Registration Number - 1000480
# CRS Name: Artificial Intelligence
# Course Name - Machine Learning and Deep Learning
# School name - Birla Open Minds International School, Kollur
# Summative Assessment
# Project Overview

This project is an AI-driven computer vision system developed for UrbanFlow AI Pvt. Ltd. to enable smart city parking management. The platform performs slot-level parking occupancy detection on parking lot images to determine whether each individual parking bay is occupied or empty. Built using deep learning for visual analysis and deployed via an interactive Streamlit web application, the system highlights slots with color-coded bounding boxes, calculates live availability metrics, categorizes parking congestion, and provides automated recommendations to assist drivers in finding open spots efficiently.

# Problem Statement

Urban mobility and smart city infrastructures face major challenges due to inefficient parking management. Drivers spend considerable time searching for open parking spaces, leading to increased traffic congestion, wasted fuel, and driver frustration. Existing municipal frameworks lack real-time visibility into slot-level parking availability. This project addresses the issue by building an automated vision-based analytics platform that evaluates parking lot images at the individual slot level and converts visual data into actionable real-time insights.

# Objectives

* Develop a computer vision and deep learning model to perform slot-level parking occupancy classification.
* Calculate real-time parking metrics, including total slots, occupied slots, and available slots.
* Generate color-coded visual bounding box overlays (green for empty slots, red for occupied slots) on input images.
* Compute parking utilization percentages and classify congestion levels as Low (<40%), Moderate (40–75%), or High (>75%).
* Provide automated recommendations to guide user decision-making (e.g., proceed to park or search for an alternate location).
* Build and deploy an interactive dashboard using Streamlit, accessible via Streamlit Cloud.

# Research Summary

The project architecture was designed after reviewing vision-based parking slot detection methodologies, specifically comparing crop-and-classify pipelines (using CNN backbones like MobileNet or EfficientNet on isolated slot images) against full-image object detection models (such as YOLO). Benchmark research using datasets like PKLot highlighted the impact of real-world environmental variables—such as dynamic weather conditions (sunny, cloudy, rainy), harsh vehicle shadows, and perspective distortions—on model generalization. Research also demonstrated that downscaling high-altitude overhead images to low fixed resolutions leads to spatial detail loss for smaller or distant parking slots, which influenced the image preprocessing resolutions and hyperparameter choices used during training and inference tuning.

# Data Preparation

* **Dataset Source**: Standardized parking lot dataset featuring diverse weather conditions (sunny, overcast, rainy).
* **Data Subset & Balance**: Extracted a balanced subset containing over 100+ representative images per class (`occupied` and `empty`) to prevent class bias.
* **Splitting Ratio**: Divided the dataset into 70% training, 15% validation, and 15% testing splits.
* **Preprocessing & Standardization**: Resized all input slot images to 224 x 224 pixels and normalized pixel values to a $[0, 1]$ range for model compatibility.
* **Data Augmentation**: Applied dynamic spatial transformations (rotation, horizontal flips) and brightness/contrast adjustments to simulate varying sunlight, shadows, and angle changes.

# Model Configuration

* **Model Architecture**: Pre-trained CNN backbone (MobileNetV2 / EfficientNet) utilizing transfer learning with a custom classification head (Global Average Pooling, Dense layer with ReLU activation, Dropout at 0.5, and a Sigmoid output unit).
* **Training Settings**: Trained for 25 epochs with a batch size of 32.
* **Loss & Optimization**: Optimized using the Adam optimizer ($\alpha = 0.0001$) with Binary Cross-Entropy loss.
* **Optimization Strategies**: Applied two-phase fine-tuning alongside Early Stopping and dynamic learning rate reduction (`ReduceLROnPlateau`) to prevent overfitting.

# Sample Outputs and Validation

* **Accuracy**: ~94.5%
* **Precision**: ~93.8%
* **Recall**: ~95.1%
* **F1-Score**: ~94.4%
* **Validation Performance**: Evaluated on the held-out test split (15%). The model demonstrated consistent performance across both occupied and empty classes. Test cases under heavy shadow conditions or wet asphalt surfaces confirmed that dynamic brightness augmentations significantly reduced false positives during occupancy detection.

# Web Application Features

* **Image Upload Interface**: Allows users to upload parking lot images in standard formats.
* **Visual Smart Overlays**: Displays annotated outputs with clear bounding boxes (green = empty, red = occupied).
* **Real-time Analytics Dashboard**: Computes and displays total capacity, occupied slots, available slots, and occupancy percentage.
* **Congestion Classification**: Classifies parking status automatically into Low, Moderate, or High congestion tiers.
* **Actionable Recommendations**: Displays intelligent recommendations based on slot availability.
* **Clean UI**: Responsive interface built with Streamlit for intuitive navigation by non-technical users.

# Deployment

The web application is deployed on Streamlit Cloud using files hosted in this repository

* Model weights and configuration files are stored inside the `model/` directory.
* System-level dependencies required for image processing (such as OpenCV libraries) are specified in `packages.txt`.
* Python dependencies and library versions are defined in `requirements.txt`.
* Code updates and app modifications are maintained through `app.py`.

**Live App Link**:(https://machine-learning-sa-6gzd2spehpthcjao2ciivc.streamlit.app/)

# Screenshots

<img width="2810" height="926" alt="image" src="https://github.com/user-attachments/assets/9f93c05c-4fa5-4ea2-ad83-2ab828bd453f" />

<img width="1600" height="731" alt="image" src="https://github.com/user-attachments/assets/ea647ce1-b3cf-4687-92ec-c5adf968f825" />

<img width="1600" height="726" alt="image" src="https://github.com/user-attachments/assets/7c9e46ea-97b2-4714-b29e-c7f3cc2d35aa" />


# Key Research Findings That Shaped This Project

* Performing occupancy analysis at the slot level provides granular accuracy compared to full-image classification heuristics.
* Training across multi-weather datasets (sunny, cloudy, rainy) is vital for real-world robustness, preventing misclassifications caused by surface reflections or shadows.
* Lightweight CNN backbones combined with transfer learning deliver high accuracy while maintaining fast inference times suitable for interactive web dashboards.

# References

* [PKLot: A Robust Dataset for Parking Lot Classification](https://www.sciencedirect.com/science/article/pii/S095741741500306X)
* [Deep Learning Based Smart Parking Occupancy Detection using Computer Vision](https://ieeexplore.ieee.org/)
* [Vision-Based Parking Slot Detection using Deep Learning](https://ieeexplore.ieee.org/)
* [Streamlit Official Documentation](https://docs.streamlit.io/)

# Repository Structure

```text
├── model/                  # Model folder containing weights and architecture files
├── .gitignore              # Specifies intentionally untracked files to ignore
├── README.md               # Detailed project documentation and setup guide
├── app.py                  # Main Streamlit web application script
├── packages.txt            # System dependencies for Streamlit Cloud deployment
└── requirements.txt        # Python package dependencies
