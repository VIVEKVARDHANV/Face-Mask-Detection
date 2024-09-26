# Realtime Face Mask Classification

## Overview

The COVID-19 pandemic has heightened the importance of wearing face masks for safety. This project addresses the need for real-time face mask detection and classification. Using advanced machine learning techniques, the goal is to develop a system that not only detects the presence of face masks but also classifies the type of mask being worn.

This project utilizes a combination of Convolutional Neural Networks (CNNs), Transfer Learning, and real-time image processing techniques to achieve robust and accurate results.

## Problem Statement

The objective is to build a robust solution that can:
1. Detect whether a person is wearing a face mask or not.
2. Identify the type of face mask being worn from the following categories:
   - N-95
   - N-95v
   - Surgical
   - Cloth
   - No Face Mask

## Dataset

The dataset used for training and testing is sourced from Kaggle: [Face Mask Types Dataset](https://www.kaggle.com/datasets/bahadoreizadkhah/face-mask-types-dataset).

- **N-95**: 354 Train, 50 Test
- **N-95v**: 390 Train, 49 Test
- **Surgical**: 342 Train, 75 Test
- **Cloth**: 396 Train, 78 Test
- **No Face Mask**: 474 Train, 78 Test

## Project Objectives

1. **Pre-Processing and Data Augmentation**: Enhance the dataset through resizing, rotation, and other transformations to create a more comprehensive training set.
2. **Model Development**:
   - **CNN Models**: Implement two CNN models with different architectures to classify face masks.
   - **Transfer Learning**: Utilize pretrained models such as ResNet-50 and MobileNet-V2 for improved accuracy and efficiency.
3. **Real-Time Detection**: Employ OpenCV with Haar cascades for face detection and classify the type of mask in real-time.

## Approach

### Phase 1: Pre-Processing and Data Augmentation
- Resize, rotate, and augment images to simulate various real-world scenarios.

### Phase 2: Model Development
- **CNN Models**:
  - **CNN-1**: A fully connected model with multiple layers.
  - **CNN-2**: A simplified model with fewer layers.
- **Transfer Learning**:
  - **ResNet-50**: Chosen for its superior accuracy and efficient memory management.
  - **MobileNet-V2**: Evaluated for its suitability for constrained environments.

### Real-Time Detection
- Use `haarcascade_frontalface_default.xml` with OpenCV for face detection.
- Classify the detected face mask type using the trained models.

## Results

- **CNN Models**: Comparison of accuracy for CNN-1 and CNN-2.
- **Transfer Learning Models**: Performance metrics for ResNet-50 and MobileNet-V2.
- **Real-Time Detection**: Demonstrated through live image processing with Streamlit.

## Deployment

The final model is deployed using [Streamlit](https://streamlit.io/) for real-time mask classification. Users can interact with the model through a web interface to test mask detection on live images.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/realtime-face-mask-classification.git
   cd realtime-face-mask-classification
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Dataset: [Face Mask Types Dataset](https://www.kaggle.com/datasets/bahadoreizadkhah/face-mask-types-dataset)
- Pretrained Models: ResNet-50 and MobileNet-V2 from TensorFlow
- Real-time face detection: OpenCV Haar cascades

## Contact

For any questions or feedback, please reach out to [your email](mailto:youremail@example.com).
