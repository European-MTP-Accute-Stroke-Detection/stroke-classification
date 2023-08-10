"""
Author: kiduk kang

Description: This script trains a patient level classifier(RNN architecture).
Patient level classifier takes features created by image level classifier as sequence.
VisionTransformer(vit_l_16)-RNN model and ResNet(resnet152)-RNN model can be trained with this script.
"""

import numpy as np
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import argparse
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

parser = argparse.ArgumentParser(description='Train patient level classifier')
parser.add_argument('--model_name', type=str, default='vit_l_16', help='input image level classifier model name, (defulat: vit_l_16)')
parser.add_argument('--batch_size', type=int, default=4, help='input batch size for training (default: 4)')
parser.add_argument('--epochs', type=int, default=500, help='number of epochs to train (default: 500)')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
parser.add_argument('--hidden_size', type=int, default=150, help='Hidden size for RNN (default: 150)')
parser.add_argument('--num_layer', type=int, default=2, help='Number of layers for (stacked)RNN (default: 2)')
parser.add_argument('--data_dir', type=str, default='/ceph/inestp02/stroke_classifier/data/patientwise/preprocess',
                    help='data directory (default: /ceph/inestp02/stroke_classifier/data/patientwise/preprocess)')
parser.add_argument('--model_dir', type=str, default='full_preprocess_14k',
                    help='model directory (default: full_preprocess_14k)')
parser.add_argument('--device', type=str, default="cuda", help='Device (default: cuda)')
parser.add_argument('--max_scan_length', type=int, default=80, help='number of maximum CT scans in a single case(default: 80)')
parser.add_argument('--tensorboard', type=bool, default=False, help='logging using tensorboard (default: False)')
args = parser.parse_args()


device = args.device if torch.cuda.is_available() else "cpu"

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
    x = [np.concatenate(
        (i, torch.zeros([maxlen-len(i)]+list(i[0].shape)))) for i in x]
    return x

#exrtact features concatenated with logit
def extract_features(images, model, device='cuda'):
    class Identity(nn.Module):
        def __init__(self):
            super(Identity, self).__init__()

        def forward(self, x):
            return x
        
    concatenated_features = []
    
    if model.__class__.__name__.find('ResNet') == 0:
        feature_extractor = nn.Sequential(*list(model.children())[:-1])
        classifier = nn.Sequential(*list(model.children())[-1])

    elif model.__class__.__name__.find('Vision') == 0:
        feature_extractor = model
        classifier = model.heads
        feature_extractor.heads = Identity()

    feature_extractor.to(device)
    classifier.to(device)

    with torch.no_grad():
        for i, imgs in enumerate(images):
            temp = []
            for img in imgs:
                img = torch.Tensor(np.transpose(img, (2, 0, 1))).unsqueeze(0)
                img = img.to(device)
                features = feature_extractor(img).squeeze()
                features = features.to(device)
                output = classifier(features)

                concat_features = torch.cat((features,output)).unsqueeze(0)
                temp.append(concat_features)

            temp = torch.cat(temp)  # list of extracted features and logit 
            concatenated_features.append(temp)
    feature_list = torch.stack(concatenated_features)
    return feature_list  # list of features, number of patient *maxlen * n_class

    
#model save path for patient level classifier
p_model_path = f'patientwise_models/{args.model_dir}/'
if not os.path.exists(p_model_path):
    os.makedirs(p_model_path)

#model path for image level classifier
i_model_path = f'imagewise_models/{args.model_dir}/'
if not os.path.exists(i_model_path):
    os.makedirs(i_model_path)

#state_dict path
p_state_dict_path = f'patientwise_models_state_dict/{args.model_dir}/'
if not os.path.exists(p_state_dict_path):
    os.makedirs(p_state_dict_path)

i_state_dict_path = f'imagewise_models_state_dict/{args.model_dir}/'
if not os.path.exists(i_state_dict_path):
    os.makedirs(i_state_dict_path)

if args.tensorboard == True:
    #tensorboard path
    tensorboard_path = f'patientwise_tensorboard/{args.model_dir}/'
    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)

    #writing log for tensorboard
    writer = SummaryWriter(log_dir=f'{tensorboard_path}{args.model_name}')
torch.cuda.empty_cache()

#define image level classifier for feature extraction
model_name = args.model_name
try:
    if model_name in ['resnet152', 'vit_l_16']:
        pass
    else:
        raise Exception

    if model_name == 'resnet152':
        img_model = models.resnet152(weights="DEFAULT")
    elif model_name == 'vit_l_16':
        img_model = models.vit_l_16(weights="DEFAULT")

except:
    print('Error: wrong model name')

num_classes = 3

#redefine fc layer
if model_name.find('resnet') == 0:
    in_features = img_model.fc.in_features
    img_model.fc = nn.Sequential(
        nn.Linear(in_features=in_features, out_features=num_classes)
    )
elif model_name.find('vit') == 0:
    in_features = img_model.heads.head.in_features
    img_model.heads = nn.Sequential(
        nn.Linear(in_features=in_features, out_features=num_classes, bias=True)
    )

img_model.load_state_dict(torch.load(f'{i_state_dict_path}{model_name}.ckpt'))
img_model.eval()

with open(f'{args.data_dir}/patient_data_combine.pickle', 'rb') as f:
    patientwise_scans = pickle.load(f)

maxlen = args.max_scan_length

#extract image and features
images = extract_img(patientwise_scans, maxlen)
features = extract_features(images, img_model)

X = features
y = torch.Tensor(list(i['patient_label'] for i in patientwise_scans.values()))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state = 42)

train_dataset = PatientwiseDataset(X_train, y_train)
val_dataset = PatientwiseDataset(X_val, y_val)
test_dataset = PatientwiseDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size)


# Define the BiGRU model
class BiGRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(BiGRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size,
                          num_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0),
                         self.hidden_size).to(x.device)  # Initial hidden state
        out, _ = self.gru(x, h0)
        # Use the last time step's output for classification
        out = self.fc(out[:, -1, :])
        return out

# Create an instance of the BiGRU model
input_size = X_train.shape[-1]
hidden_size = args.hidden_size
num_layers = args.num_layer
output_size = 3

#Define rnn model
RNN_model = BiGRUModel(input_size, hidden_size, num_layers, output_size)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(RNN_model.parameters(), lr=args.lr)

num_epochs = args.epochs
RNN_model.to(device)

#set initial values for earlystopping
best_model = None
best_val_loss = np.inf
patience = 5
patience_check = 0

for epoch in range(num_epochs):
    RNN_model.train()
    train_loss = 0.0
    train_acc = 0.0
    for i, (features, labels) in enumerate(train_loader):
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = RNN_model(features)

        # Compute loss
        loss = criterion(outputs, labels.type(torch.long))

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        #loss
        train_loss += loss.item()*features.size(0)
        _, predicted = torch.max(outputs.data, 1)
        train_acc += (predicted == labels).sum().item()

    # Print loss and accuracy for epoch
    train_loss /= len(train_loader.dataset)
    train_acc /= len(train_loader.dataset)

    if args.tensorboard == True:
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Acc/train", train_acc, epoch)   
    print('Epoch [{}/{}], Train Loss: {:.4f}, Train Accuracy: {:.4f}'.format(
        epoch+1, num_epochs, train_loss, train_acc))

    #validation
    RNN_model.eval()
    val_loss = 0.0
    val_acc = 0.0

    with torch.no_grad():
        for i, (features, labels) in enumerate(val_loader):
            features = features.to(device)
            labels = labels.to(device)
            # Forward pass
            outputs = RNN_model(features)
            loss = criterion(outputs, labels.type(torch.long))

            # Update loss and accuracy
            val_loss += loss.item() * features.size(0)
            _, predicted = torch.max(outputs.data, 1)
            val_acc += (predicted == labels).sum().item()

        # Compute average loss and accuracy for epoch
        val_loss /= len(val_loader.dataset)
        val_acc /= len(val_loader.dataset)
        if args.tensorboard == True:
            writer.add_scalar("Loss/validation", val_loss, epoch)
            writer.add_scalar("Acc/validation", val_acc, epoch)
        print('Epoch [{}/{}], Validation Loss: {:.4f}, Validation Accuracy: {:.4f}'.format(
            epoch+1, num_epochs, val_loss, val_acc))

        #early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_check = 0
            best_model = RNN_model

            #save model state_dict
            torch.save(best_model.state_dict(),
                       f'{p_state_dict_path}{model_name}_rnn.ckpt')
            #save entire model
            torch.save(best_model, f'{p_model_path}{model_name}_rnn.ckpt')

        else:
            patience_check += 1
        if patience_check >= patience:
            break

#Test
RNN_model.load_state_dict(torch.load(f'{p_state_dict_path}{model_name}_rnn.ckpt'))

RNN_model.eval()
test_loss = 0.0
test_acc = 0.0
predicted_labels = []
true_labels = []

with torch.no_grad():
    for i, (features, labels) in enumerate(test_loader):
        features = features.to(device)
        labels = labels.to(device)
        #Forward pass
        outputs = RNN_model(features)
        loss = criterion(outputs, labels.type(torch.long))

        # Update loss and accuracy
        test_loss += loss.item() * features.size(0)
        _, predicted = torch.max(outputs.data, 1)
        test_acc += (predicted == labels).sum().item()
        
        #store predicted and true labels
        predicted_labels.extend(predicted.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

# Compute average loss and accuracy for epoch
test_loss /= len(test_loader.dataset)
test_acc /= len(test_loader.dataset)
if args.tensorboard == True:
    writer.add_scalar("Loss/test", test_loss)
    writer.add_scalar("Acc/test", test_acc)
print('Test Loss: {:.4f}, Test Accuracy: {:.4f}'.format(
    test_loss, test_acc))

#classification_report
print("Classification report")
print(classification_report(true_labels, predicted_labels))

# Calculate and print confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(true_labels, predicted_labels))

if args.tensorboard == True:
    writer.flush()
    writer.close()
