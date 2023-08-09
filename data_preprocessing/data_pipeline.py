"""
Author: Eunmi Joo

Data Pipeline

Load RSNA, AISD dataset (dicom format, patient-wise)
    - data/patientwise/raw/patient_data_hemo_1200.pickle
    - data/patientwise/raw/patient_data_isch.pickle
    - data/patientwise/raw/patient_data_neg_1200.pickle

Save preprocessed combined dataset
    - data/patientwise/preprocess/patient_data_combine_full.pickle"
"""

import pydicom, numpy as np
import matplotlib.pylab as plt
import pickle
import scipy.ndimage as ndi
import math
import cv2
from skimage import morphology
import torch
import os
from tqdm.notebook import tqdm
device = "cuda:3" if torch.cuda.is_available() else "cpu"
torch.cuda.set_device(device)
import warnings
from data_preprocess import data_preprocess
warnings.filterwarnings("ignore")

# load patient-wise data - hemorrhage, ischemic
root = "/ceph/inestp02/stroke_classifier/data/patientwise/raw/"
hemo_path = "patient_data_hemo_1200.pickle"
isch_path = "patient_data_isch.pickle"
neg_path = "patient_data_neg_1200.pickle"

with open(root+hemo_path, "rb") as f:
    patient_data_hemo = pickle.load(f)

with open(root+isch_path, "rb") as f:
    patient_data_isch = pickle.load(f)

with open(root+neg_path, "rb") as f:
    patient_data_neg = pickle.load(f)
    
combine_data = dict(**patient_data_hemo, **patient_data_isch, **patient_data_neg)

root = "/ceph/inestp02/stroke_classifier/data/patientwise/preprocess/"
path = "patient_data_combine_full.pickle"

patient_data = combine_data
patient_data_mv = dict()
for patient in tqdm(patient_data.keys()):
    patient_data_mv[f'{patient}_1'] = dict()
    patient_data_mv[f'{patient}_2'] = dict()
    patient_data[patient]['img_preprocess'] = []
    for dicom in patient_data[patient]['dicom']:
        _, img_preprocess, is_abnormal = data_preprocess(dicom) # applying data preprocessing
        patient_data[patient]['img_preprocess'].append(img_preprocess)
    
    pos2 = patient_data[patient]['pos2']
    pos2 = np.array(pos2)
    min_pos = np.min(pos2)
    max_pos = np.max(pos2)
    pos2_norm = (np.array(pos2)-min_pos)/(max_pos-min_pos)
    sort_idx = np.argsort(pos2_norm)
    rev_sort_idx = sort_idx[::-1]
    
    # store attr items of patient-wise dataset sorted, reverse-sorted in pos2 (CT scans z pos)
    # to add dataset and robust in direction of the brain CT scans
    for item in patient_data[patient].keys():
        if item == 'patient_label' or item == 'dicom':
            continue
        item_arr = np.array(patient_data[patient][item])
        patient_data_mv[f'{patient}_1'][item] = item_arr[sort_idx]
        patient_data_mv[f'{patient}_2'][item] = item_arr[rev_sort_idx]
    patient_data_mv[f'{patient}_1']['patient_label'] = patient_data[patient]['patient_label']
    patient_data_mv[f'{patient}_2']['patient_label'] = patient_data[patient]['patient_label']
    # patient_data_mv attr: ['pos2', 'image_label', 'img_preprocess', 'patient_label']

with open(root+path, "wb") as f:
    pickle.dump(patient_data_mv, f, pickle.HIGHEST_PROTOCOL)
