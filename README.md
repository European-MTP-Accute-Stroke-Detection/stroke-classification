# Stroke Type Classifier Model Implementation

This repository contains the implementation code for a Stroke Type Classifier model. The model is designed to classify different types of strokes based on medical imaging data, such as MRI or CT scans. Accurate classification of stroke types is crucial for timely and effective medical intervention.

![Model Architecture](./assets/model_architecture.png)

## Table of Contents

- [Introduction](#introduction)
- [Repository Structure](#repository-structure)
- [Data and Data Preprocessing](#data)
- [Model](#model)
- [License](#license)
- [Contact](#contact)

## <a name="introduction"></a>Introduction

Stroke classification for accurate brain stroke detection software that can support medical practitioners in diagnosis of stroke in patients.

## <a name="repository-structure"></a>Repository Structure

- `data_preprocessing/`: Code for data preprocessing and augmentation.
	- 'image_to_patient.py': build patient-wise dataset
	- 'balance_data.ipynb': balance the data volume to handle data imbalance
	- 'data_preprocess.py': functions for data preprocessing
	- 'data_pipeline.py': run data preprocessing and augmentation
	- 'compute_image_feature.py': compute multi-layer images into features with fine-tuned image-level model
	
- `model/`: Implementation of the stroke type classifier model architecture.
	- 'patient_classifier_transformer.py': patient-wise stroke type classifier model implementation - Transformer Architecture
   	- 'model_selection.ipynb': model selection scripts - hyperparameter optimization

- `requirements.txt`: List of required Python libraries.
- `LICENSE`: Information about the open-source license.
- `README.md`: The document you are currently reading.

## <a name="data"></a>Data and Data Preprocessing

It uses the RSNA and AISD dataset to gain various types of stroke brain CT scans including negative, hemorrhage, and ischemic.
You'll first have to download the datasets in order to run the notebooks. Additional labelling to the dataset is required as both dataset are for different purpose - hemorrhage type classification and ischemic area segmentation.

![Data Pipeline](./assets/data_pipeline.png)

### Download Links

RSNA(Hemorrhage & Negative): https://www.kaggle.com/competitions/rsna-intracranial-hemorrhage-detection/overview
AISD(Ischemic): https://github.com/GriffinLiang/AISD

### Data Preprocessing

Data Preprocessing to gain more consistent input for the model despite of different sources of the brain CT scans.

![Preprocessing](./assets/preprocessing.png)

### Data Result

![Preprocessing Result](./assets/preprocessing_result.png)

## <a name="model"></a>Model

We implemented image-level classifier and patient-level classifier.

- Image-level Classifier: classify stroke type of one brain CT scan.
- Patient-level Classifier: classify stroke type of one patient based on the whole brain CT scans of the patient.

### Patient-level Classifier Architecture

We implemented patient-level stroke classifier in two different approaches - RNN and Transformer.

### RNN

![RNN Architecture](./assets/rnn_architecture.png)
![RNN Result](./assets/rnn_result.png)

### Transformer

![Transformer Architecture](./assets/transformer_architecture.png)
![Transformer Result](./assets/transformer_result.png)

## <a name="license"></a>License
Distributed under the MIT License.

## <a name="contact"></a>Contact

[Eunmi Joo's Github](https://github.com/eunmi228)

[Kiduk Kang's Github](https://github.com/kdkangg)
