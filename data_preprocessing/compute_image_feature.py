"""
Author: Eunmi Joo

Compute multi-layer images into features with fine-tuned image-level classifier model

Load RSNA, AISD patient-wise dataset and image-level classifier model
    - data/patientwise/preprocess/patient_data_combine_wo_dicom.pickle
    - models_state_dict/full_preprocess_14k/resnet152.ckpt
    
Save image's features computed by image-level classifier
    - data/patientwise/preprocess/features_resnet152.pickle
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import numpy as np

from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter


import pickle
from sklearn.metrics import classification_report, confusion_matrix

import os
import argparse
import pickle

device = "cuda:4" if torch.cuda.is_available() else "cpu"
torch.cuda.set_device(device)

class PatientwiseDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y

#extract sequence image data from the data and pad it to the max length
def extract_img(data, maxlen):
    x = [data[i]['img_preprocess'] for i in list(data.keys())]
    x = [np.concatenate((i, torch.zeros([maxlen-len(i)]+list(i[0].shape)))) for i in x]
    return x

def extract_logit(images, model, device='cuda'):
    logit_list = []
    model.to(device)
    with torch.no_grad():
        for i, imgs in tqdm(enumerate(images)):
            temp = []
            for img in imgs:
                img = torch.Tensor(np.transpose(img, (2, 0, 1))).unsqueeze(0)
                img = img.to(device)
                output = model(img)
                temp.append(output)

            temp = torch.cat(temp)  # list of probabilities in each patient
            logit_list.append(temp)
    logit_list = torch.stack(logit_list)
    return logit_list  # list of logits, number of patient *maxlen * n_class

with open('/ceph/inestp02/stroke_classifier/data/patientwise/preprocess/patient_data_combine_wo_dicom.pickle', 'rb') as f:
    patientwise_scans = pickle.load(f)
    
model = models.resnet152(weights="DEFAULT")
in_features = model.fc.in_features
model.fc = nn.Sequential(
        nn.Linear(in_features = in_features, out_features = 3)
    )
model.load_state_dict(torch.load('/ceph/kikang/MTP/models_state_dict/full_preprocess_14k/resnet152.ckpt'))
model.eval()

maxlen = 80

# Set the device for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

images = extract_img(patientwise_scans, maxlen)

model_wo_fc = nn.Sequential(*list(model.children())[:-1])
logits = extract_logit(images, model_wo_fc)
with open('/ceph/inestp02/stroke_classifier/data/patientwise/preprocess/features_resnet152.pickle', 'wb') as f:
    pickle.dump(logits, f)
