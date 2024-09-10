# Facial Extraction Project 🚀

## Description 📖
This project focuses on developing a facial feature extraction system using deep learning models. It integrates face detection and feature extraction to facilitate advanced applications like facial recognition, gender classification, and more. 

Method: use 2 separate models (YOLOV8s for head/face detection and ResNet50 for classification, extracting attributes) with the purpose of being able to evaluate 2 models independently to be able to Improve and optimize without affecting each other

## Table of Contents 🗂️
- [Project Description](#description)
- [Data Collection](#data-collection)
- [Model Training](#model-training)
- [Model Features](#model-features)
- [Results](#results)
- [Installation](#installation)
- [License](#license)

## Data Collection 📊
### Data Sources
Source for Detection: We use a raw data set collected from sources on Kaggle including 10,000 daily life photos. The images have stable resolution, there are few blurred or noisy images and almost over 90% of the images can be used for 'labeling'.  

Source for Classification:  We use a raw data set collected from sources on Kaggle including 10,000 daily life photos. The images have stable resolution, there are few blurred or noisy images and almost over 90% of the images can be used for 'labeling'.  

### Data Labeling 🏷️
To make the data set as complete as possible, we have filtered out images with poor resolution, poor quality, taken at the wrong location, or images without human subjects. The labeling tool used is Roboflow.

Data for Detection: After completing the labeling process, it is estimated that 7,376 images have been labeled (resized to 640x640) with a total of 35,310 images.

Data for Classification: With the data for the Resnet50 model, the statistics show that it was done to label 7233 images (resized to 224x224).

### Data Augmentation
With the data for the head dectetion problem, we performed data augmentation techniques directly on Roboflow. 

Data for Detection: The team obtained a larger data set of up to 17,000 images for the training set and 2000 images for the validation set. Includes Flip: Horizontal, Crop: 0% Minimum Zoom, 30% Maximum Zoom, Exposure: Between -16% and +16%, Bounding Box: Blur, Noise, Rotation

Data for Classification: After performing data augmentation techniques directly on Roboflow, we have obtained a larger data set of up to 15,177 images for the training set, 2174 images for the validation set, and 600 images for the test set.

## Model Training ⚙️
### Environment
2 models were trained separately by using GPU from Google Colab. The first one YOLOv8s was trained using Pytorch while ResNet using Keras. 

### Models

#### YOLOv8s for Face Detection
The YOLO (You Only Look Once) series is well-known for its efficiency and accuracy in object detection tasks. We chose YOLOv8s, the latest iteration in the YOLO series, for face detection due to its enhanced performance and speed. YOLOv8s is designed to be incredibly fast, making it ideal for real-time applications. It achieves high accuracy while being able to detect faces in various lighting conditions and angles, which is crucial for the robustness of our facial extraction system.

#### ResNet50 for Feature Extraction
ResNet50 is part of the Residual Network family, which is famous for its deep architecture that effectively addresses the vanishing gradient problem through the use of skip connections. We selected ResNet50 due to its powerful feature extraction capabilities. This model can capture intricate details from detected faces, providing a rich set of features necessary for subsequent analysis tasks such as facial recognition or emotion detection. ResNet50's ability to learn from a large amount of data and its generalization over different facial attributes make it an excellent choice for our project.

#### Why These Models Were Chosen
The combination of YOLOv8s and ResNet50 offers a balance of speed and accuracy, which is essential for deploying a practical facial recognition system. YOLOv8s provides fast and reliable face detection, which is critical for real-time processing scenarios. Meanwhile, ResNet50 complements this by delivering detailed and comprehensive feature extraction, enabling accurate identification and classification of facial features. This synergy ensures our system not only detects faces quickly but also analyzes them with high precision, making it suitable for a variety of applications from security systems to marketing analysis.

### Training Process 🛠️
Training YOLOv8s: Image_size's Input: 640x640, batch_size: default, epoch: 50, data was split into 2 main parts: 85% for training and 15% for validation.

Training ResNet50: Image_size's Input: 224x224, model was pre-trained on ImageNet dataset, which are not really suitable for human (face) task but there are some features that might be useful for our problem. Batch_size: 32, epoch: 50, optimizer: Adam and we used Early Stopping with val_loss, patience = 5 in order to avoid being 'overfitting'. Data was split into 2 main parts: 85% for training and 15% for validation.

## Model Architecture 🏛️

### YOLOv8s Modifications
The YOLOv8s model in our project has been fine-tuned to enhance its efficiency in face detection tasks. We adjusted the input size to better accommodate the typical dimensions of faces in our dataset, optimizing the model to focus more precisely on facial features. Additionally, we customized the anchor boxes to better match the aspect ratios and scales of faces in diverse conditions, which helps improve detection accuracy.

### ResNet50 Modifications
For ResNet50, modifications were made to adapt the model to the specific needs of feature extraction from facial images. We incorporated additional convolutional layers to deepen the network, aiming to capture more detailed facial features essential for subsequent analysis tasks. Furthermore, dropout layers were added to prevent overfitting, maintaining the model's generalization capabilities across different facial datasets.

### Integration
The outputs from the YOLOv8s face detection are directly fed into the modified ResNet50 model for feature extraction. This integrated approach ensures seamless operation, where face detection and feature extraction are optimized to work in tandem, providing a robust system for accurate facial analysis.

## Results 📈
### Model Evaluation
#### Evaluate YOLOv8s: 

Train loss and Val loss decreased steadily and gradually converges as the epoch increases while accurcay in both training set, validation set were increasing. 
Recall: 0.8, Precision: 0.863, mAP50: 0.871, mAP50-95: 0.533.

#### Evaluate ResNet50: 

Accuracy during training is stable at nearly 100%, this is a quite high number. However, the testing accuracy is lower at approximately 95% but has more fluctuations, which shows that the model may be overfitting to the training data. Val_acc: 0.95

Precision: The model shows high precision on training data but exhibits lower precision and variability on testing data. Val_Precision: 0.92

Recall: Recall is high for training but drops for validation data, suggesting issues with generalization. Val_Recall: 0.95

Loss: Training loss decreases sharply and stabilizes, while validation loss trends upwards after initial decrease, indicating overfitting. Val_loss: 0.22

AUC: AUC values are high for both training and validation, demonstrating strong classification ability despite other issues. Val_AUC: 0.97


## Installation 🔧
Run step by step: 

Clone data from Roboflow, set up GPU/CPU, run all cells code in file 'Head Detection', save model_yolov8s.pt

Clone data from Roboflow, set up GPU/CPU, run all cells code in file 'ResNet50', save model_resnet50.h5

Run all cells code in file 'Connection' in order to connect 2 models so that you will have a perfect model for further custom

## License ⚖️
Feel free to customize this template further based on the specifics of your project and the technologies used. This structured format helps ensure that all relevant aspects of your project are clearly communicated to potential users and contributors.



