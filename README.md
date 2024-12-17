# SRNet_CBAM - Steganalysis Model with CBAM Attention Mechanism

## Overview
SRNet_CBAM is an advanced steganalysis model that builds upon the SRNet architecture by integrating the Convolutional Block Attention Module (CBAM). This enhancement aims to improve the model's performance in detecting hidden messages within digital media, such as images.

## Features
- **Enhanced Detection Capability**: The incorporation of CBAM allows the model to focus on more relevant features and patterns within the stego-images, leading to more accurate detection of hidden data.
- **Advanced Attention Mechanism**: CBAM computes attention maps along both spatial and channel dimensions, enabling the model to selectively emphasize important regions and feature maps, thus enhancing its discriminative power.
- **Robust Architecture**: Based on the reliable SRNet foundation, SRNet_CBAM inherits its robustness and adaptability to various steganography techniques, making it a versatile tool for steganalysis tasks.

## Installation
1. Clone the repository:
```bash
git clone https://github.com/Ephemeral2333/SRNet_CBAM.git
```
2. Install the required dependencies. The project likely requires popular deep learning frameworks such as TensorFlow or PyTorch, along with other common libraries like NumPy and OpenCV. You can install them using `pip` or `conda`. For example, if using `pip`:
```bash
pip install -r requirements.txt
```

## Usage
1. **Data Preparation**: Prepare your stego-images and cover-images datasets in a suitable format. The dataset should be split into training, validation, and testing subsets. The dataset used in our experiments can be downloaded from [here](https://drive.google.com/file/d/10OSYnkWYLF166zJOk_x6Olo5IR150l9U/view?usp=drive_link). Please note that you may need to adapt the dataset to the specific requirements of the model and the scripts provided.
2. **Model Training**: Use the provided training scripts to train the SRNet_CBAM model on your dataset. You may need to adjust hyperparameters such as learning rate, batch size, and number of epochs according to your specific requirements.
```bash
python train.py
```
3. **Model Evaluation**: After training, evaluate the performance of the model using the testing subset of your dataset. The evaluation metrics such as accuracy, precision, recall, and F1-score will be calculated to assess the model's effectiveness.
```bash
python test_model.py
```
4. **Prediction**: Use the trained model to predict whether an image contains hidden data or not.
```bash
python predict_image.py
```
## Results
Our experiments on standard steganalysis datasets have shown that SRNet_CBAM outperforms the original SRNet model, achieving higher detection accuracy and better generalization ability. The detailed experimental results, including comparison tables and graphs, can be found in the `results` directory of the project.