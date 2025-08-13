# Animal Image Classification using MobileNetV2

This project is a machine learning system that can automatically identify animals from images.  
It uses **MobileNetV2**, a lightweight deep learning model, to classify images into different animal categories with high accuracy.

---

## Overview

With this tool, you can:
- Upload an image of an animal and get an instant prediction of its species.
- Train the model on your own dataset of animals.
- Understand the performance of the model through accuracy and loss graphs.

**Example Use Case:**
- Wildlife researchers can quickly label large collections of animal photos.
- Zoos or wildlife apps can integrate this system for educational purposes.
- Hobbyists can experiment with AI-based image recognition.

---

## How It Works

1. **Image Input** → The system accepts an image file (JPEG, PNG, etc.).
2. **Preprocessing** → The image is resized to 224x224 pixels and normalized.
3. **Prediction** → MobileNetV2 processes the image and outputs the predicted animal category.
4. **Output** → The top prediction (and probability) is shown to the user.

---

## Example Predictions

<img width="1919" height="958" alt="image" src="https://github.com/user-attachments/assets/cf4f9048-b7e2-4382-b68e-36b3d244789b" />

<img width="1919" height="952" alt="image" src="https://github.com/user-attachments/assets/3696f07f-5230-4036-bd45-0a1547be056d" />

---

## Project Structure

```
├── model/ # Saved model files
├── dataset/ # Sample test images
├── project1.ipynb # Training and experimentation
├── requirements.txt # Dependencies
└── README.md # Documentation
```
---

## Getting Started

1. **Clone the repository:**
   ```
   git clone https://github.com/username/animal-classification.git
   cd animal-classification
   ```
2. Install dependencies:
    ```
    pip install -r requirements.txt
    ```

To train:
Run all the cells of Jupyter notebook (project1.ipynb). (Training may take some time - ~15 min)

You can adjust:

- Number of epochs
- Learning rate
- Dataset path

Model Performance
The model achieved high accuracy after fine-tuning:

Metric Value
- Training Accuracy	96%
- Validation Accuracy	95%
