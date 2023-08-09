"""
Author: Eunmi Joo

Build Patient-wise Dataset (dictionary)

Key for patient: "PatientID"_"StudyInstanceUID" (same patient could have different caseid)
    "dicom":          list of dicom file for the patient (ascending order of pos2)
    "pos2":           list of pos2(z-position) for corresponding dicom image
    "image_label":    list of label for corresponding dicom image (0: neg, 1: hemorrhage, 2: ischemic)
    "patient_label":  patient-wise label

Load image from RSNA(hemorrhage) and AISD(ischemic) dataset
"""

import torch
import os
device = "cuda" if torch.cuda.is_available() else "cpu"
import pickle
import numpy as np
from tqdm.notebook import tqdm
import pydicom
import pandas as pd

# load RSNA (hemorrhage) train dataset
root = "/ceph/inestp02/stroke_classifier/data/rsna-intracranial-hemorrhage-detection/stage_2_train/"
file_list = os.listdir(root)

# load RSNA train label
label_rsna = pd.read_csv('/ceph/inestp02/stroke_classifier/data/rsna-intracranial-hemorrhage-detection/stage_2_train.csv')
label_list_df = pd.DataFrame(columns = ['ImageID', 'label'])

# label processing from hemorrhage subtypes to one label for hemorrhage(pos = 1, neg = 0)
label_rsna['Sub_type'] = label_rsna['ID'].str.split("_", n = 3, expand = True)[2]
label_rsna['ImageID'] = label_rsna['ID'].str.split("_", n = 3, expand = True)[1]
image_label = label_rsna.groupby(['ImageID'], as_index = False).sum()
image_label.loc[image_label['Label'] != 0, 'Label'] = 1
image_label['ImageID'] = 'ID_'+image_label['ImageID']

# build hemorrhage patientwise datset
print("build hemorrhage patientwise datset")
patient_data_hemo = dict()
for file in tqdm(file_list):
    dicom = pydicom.read_file(root+file)
    key = str(dicom.PatientID)+"_"+str(dicom.StudyInstanceUID)
    imageid = dicom.SOPInstanceUID
    label = int(image_label[image_label['ImageID']==imageid]['Label'])
    if key in patient_data_hemo.keys():
        patient_data_hemo[key]['dicom'].append(dicom)
        patient_data_hemo[key]['pos2'].append(dicom.ImagePositionPatient[2])
        patient_data_hemo[key]['image_label'].append(label)
    else:
        patient_data_hemo[key] = dict()
        patient_data_hemo[key]['dicom'] = [dicom]
        patient_data_hemo[key]['pos2'] = [dicom.ImagePositionPatient[2]]
        patient_data_hemo[key]['image_label'] = [label]
        
# load AISD train label
root = "/ceph/inestp02/stroke_classifier/data/Ischemic/"
label_ischemic = pd.read_csv(root+"ischemic_label.csv")
label_ischemic.loc[label_ischemic['Label'] == 1,'Label'] = 2

# build ischemic patientwise datset
print("build ischemic patientwise datset")
data_dir_path = root+"/data/"
label_list_df = pd.DataFrame(columns = ['ImageID', 'label'])
data_dicom = []
patients_list = os.listdir(data_dir_path)
patient_data_isch = dict()
for patient in tqdm(patients_list):
    for file_name in os.listdir(data_dir_path+patient):
        dicom = pydicom.read_file(data_dir_path+patient+f"/{file_name}")
        key = "ID_"+str(dicom.PatientID)+"_"+str(dicom.StudyInstanceUID)
        imageid = dicom.SOPInstanceUID
        file_num = file_name.split(".")[0]
        label=label_ischemic.loc[label_ischemic["Patient"]==int(patient)][label_ischemic["Mask"] == int(file_num)]['Label'].item()
        if key in patient_data_isch.keys():
            patient_data_isch[key]['dicom'].append(dicom)
            patient_data_isch[key]['pos2'].append(dicom.ImagePositionPatient[2])
            patient_data_isch[key]['image_label'].append(label)
        else:
            patient_data_isch[key] = dict()
            patient_data_isch[key]['dicom'] = [dicom]
            patient_data_isch[key]['pos2'] = [dicom.ImagePositionPatient[2]]
            patient_data_isch[key]['image_label'] = [label]

# sort by pos2 (z pos)
for key in patient_data_hemo.keys():
    pos2 = np.array(patient_data_hemo[key]['pos2'])
    label = np.array(patient_data_hemo[key]['image_label'])
    dicom = np.array(patient_data_hemo[key]['dicom'])
    sort_idx = np.argsort(pos2)
    patient_data_hemo[key]['pos2'] = pos2[sort_idx]
    patient_data_hemo[key]['image_label'] = label[sort_idx]
    patient_data_hemo[key]['dicom'] = dicom[sort_idx]
    if np.sum(patient_data_hemo[key]['image_label']) == 0:
        patient_data_hemo[key]['patient_label'] = 0
    else:
        patient_data_hemo[key]['patient_label'] = 1
        
for key in patient_data_isch.keys():
    pos2 = np.array(patient_data_isch[key]['pos2'])
    label = np.array(patient_data_isch[key]['image_label'])
    dicom = np.array(patient_data_isch[key]['dicom'])
    sort_idx = np.argsort(pos2)
    patient_data_isch[key]['pos2'] = pos2[sort_idx]
    patient_data_isch[key]['image_label'] = label[sort_idx]
    patient_data_isch[key]['dicom'] = dicom[sort_idx]
    if np.sum(patient_data_isch[key]['image_label']) == 0:
        patient_data_isch[key]['patient_label'] = 0
    else:
        patient_data_isch[key]['patient_label'] = 2

# save as pickle
with open('/ceph/inestp02/stroke_classifier/data/patientwise/raw/patient_data_hemo.pickle', 'wb') as f:
    pickle.dump(patient_data_hemo, f, pickle.HIGHEST_PROTOCOL)
with open('/ceph/inestp02/stroke_classifier/data/patientwise/raw/patient_data_isch.pickle', 'wb') as f:
    pickle.dump(patient_data_isch, f, pickle.HIGHEST_PROTOCOL)