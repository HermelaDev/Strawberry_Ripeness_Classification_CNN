<p align="center">
  <img src="strawberry.jpg" alt="Sleep Health ML Banner" width="500" style="border-radius:10px;">
</p>

# Strawberry_Ripeness_Classification_CNN

## Project Overview
This project focuses on **classifying strawberries** into three categories: **Occluded, Ripe, and Unripe** using **Convolutional Neural Networks (CNNs)** and **MobileNetV2**. The goal is to build an automated system for assessing strawberry ripeness, which can assist in precision agriculture, automated harvesting, and quality control.

The dataset contains 3,367 images with a class imbalance, making it necessary to consider class weights or data augmentation during training.

---

## Features
- Preprocessing pipeline including label remapping, train/test split, normalization, and resizing images to 224x224.  
- Class imbalance handling using **class weights** and **manual data augmentation**.  
- Two model approaches:  
  1. Custom CNN  
  2. Pretrained MobileNetV2 with fine-tuning.  
- Evaluation using accuracy metrics and confusion matrices.

---

## Requirements
- Python 3.9+  
- TensorFlow 2.x  
- NumPy  
- scikit-learn  
- Matplotlib  
- Seaborn  

Install dependencies via pip:

```bash
pip install tensorflow numpy scikit-learn matplotlib seaborn
```

## Usage

1. Download the dataset:
```bash
import urllib.request

url = "https://pages.scinet.utoronto.ca/~ejspence/strawberries.npz"
urllib.request.urlretrieve(url, "strawberries.npz")
print("Download complete!")
```

2. Load and preprocess the data:

- Remap labels from {1,2,3} → {0,1,2}
- Normalize images to [0,1]
- Split into training (80%) and test (20%) sets with stratification

3. Train the model:

- Custom CNN:

```bash
# Define CNN architecture
# Compile, fit with early stopping, evaluate
```

- MobileNetV2:

```bash
# Build MobileNetV2
# Compile, fit with early stopping, evaluate
```

4. Evaluate performance:

- Check accuracy and confusion matrix on the test set
- Compare unbalanced training vs balanced (class weights or data augmentation)

## Results
Model / Approach	Training Accuracy	Test Accuracy
Custom CNN (unbalanced)	91.58%	88.13%
Custom CNN (class weights)	84.81%	88.28%
Custom CNN (augmented)	67.08%	88.72%
MobileNetV2 (class weights)	90.30%	87.83%

Confusion matrices show that balancing improves minority class detection (Occluded, Ripe) but may reduce overall training accuracy.

## Conclusion

This project demonstrates the trade-off between overall accuracy and fairness across classes in imbalanced datasets. While unbalanced training yields high accuracy, balanced approaches improve detection of minority classes, which is critical for practical applications. MobileNetV2 with transfer learning offers a robust and efficient alternative.

## License

This project is licensed under the MIT License.

## Contact
Author: Hermela Seltanu Gizaw

Email: hermelaselt@gmail.com
