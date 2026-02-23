# Project-5_Finding_Manufacture_Defects_using_CGAN
## Group Project
- Team Members: Likitha, MD Lathif, Sruthika, Navadeep, Hari Priya

---

## Overview

This project focuses on detecting manufacturing defects using a Conditional Generative Adversarial Network (CGAN).

In manufacturing industries, detecting defects such as cracks, scratches, dents, and surface abnormalities is essential for maintaining product quality. Manual inspection methods are slow, costly, and prone to human error. This project solves that problem by learning the distribution of normal (defect-free) manufacturing images and generating condition-specific normal images.

The generated images are compared with real inspection images to identify deviations, which indicate potential defects. This helps automate quality inspection and improve reliability.

---

## Project Objectives

- Learn the distribution of normal manufacturing images
- Generate realistic condition-based manufacturing images
- Detect defects using CGAN
- Reduce manual inspection effort
- Improve quality control accuracy
- Provide reusable CGAN defect detection framework

---

## Training Workflow

1. Load manufacturing dataset
2. Preprocess images (resize, normalize)
3. Initialize Conditional Generator and Discriminator
4. Train CGAN using adversarial learning
5. Generate normal images
6. Save trained Generator model
7. Use model for defect detection

---

## Project Structure

Project5_Finding_manufacture_Defects/

│
├── src/
│   ├── config.py
│   ├── deployment.py
│   ├── evaluation_metrics.py
│   ├── gan_architecture.py
│   ├── model_pipeline.py
│   ├── monitoring.py
│   └── training_pipeline.py
│
├── outputs/
│   ├── generated_images/
│   ├── checkpoints/
│   └── loss_graphs/
│
├── main.py
│
├── README.md
│
└── .gitignore

---

## Module-Wise Description

### 🔹 Module 1 — Configuration  
Likitha

- Define hyperparameters
- Training settings
- Model configuration

---

### 🔹 Module 2 — Model Design  
Likitha

### Generator

- Input: Noise vector + Label
- Generates manufacturing images
- Uses convolution layers

### Discriminator

- Input: Image + Label
- Classifies real or fake images

### Loss & Optimization

- Binary Cross Entropy Loss
- Adam Optimizer
- Adversarial training

---

### 🔹 Module 3 — Training Pipeline  
Hari Priya

- Train Generator and Discriminator
- Perform adversarial learning
- Save trained model

---

### 🔹 Module 4 — Evaluation & QA  
MD Lathif

- Compare generated and real images
- Detect defects
- Evaluate model performance

---

### 🔹 Module 5 — Deployment Layer  
Sruthika

- Load trained model
- Generate images
- Enable defect detection

---

### 🔹 Module 6 — Monitoring & Updates  
Navadeep

- Monitor loss values
- Track performance
- Save checkpoints

---

## Dataset

- https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database

Manufacturing defect dataset containing product surface images.

Dataset includes normal and defective images.

Training is done using only normal images.

---

## Technologies Used

- Python
- PyTorch
- NumPy
- Matplotlib
- Conditional GAN (CGAN)

---

# Results

The CGAN model successfully generated realistic manufacturing images that closely resemble defect-free product surfaces. The model learned normal image patterns effectively and helped identify defects by detecting deviations.

This system improves inspection accuracy and reduces manual effort.

---

# Conclusion

The project provides an efficient AI-based solution for manufacturing defect detection using CGAN. The system learns normal image distribution and detects defects automatically, improving quality control and reliability.

The modular design allows easy deployment and scalability for real-world manufacturing applications.

---
