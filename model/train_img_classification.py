"""
Author: kiduk kang

Description: This script trains a image level classifier
Image level classifier uses transfer learning approach from pretrained models.
Model families of resnet, resnext, efficientnet, visiontransformer, swintransformer can be trained with this script.
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

parser = argparse.ArgumentParser(description='Train image level classifier model')
parser.add_argument('--model_name', type=str, default='vit_l_16', help='input model name, (defulat: vit_l_16)')
parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=500, help='number of epochs to train (default: 500)')
parser.add_argument('--lr', type=float, default=1e-6, help='learning rate (default: 1e-6)')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 (default: 0.9)')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 (default: 0.999)')
parser.add_argument('--data_dir', type=str, default='data/14k_preprocessed',
                    help='data directory (default: data/14k_preprocessed)')
parser.add_argument('--model_dir', type=str, default='runs',
                    help='directory specified for model or runs (default: runs)')
parser.add_argument('--epsilon', type=float, default=1e-8, help='epsilon (default: 1e-8)')
parser.add_argument('--weight_decay', type=float, default=1e-7, help='weight_decay (default: 1e-7)')
parser.add_argument('--device', type=str, default="cuda", help='Device (default: cuda)')
parser.add_argument('--tensorboard', type=bool, default=False, help='logging using tensorboard (default: False)')
args = parser.parse_args()

class ImgwiseDatset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        x = np.transpose(x, (2, 0, 1))
        y = y.squeeze()
        return x, y
    
#load data saved in pickle
with open(f'{args.data_dir}/X.pickle', 'rb') as f:
    preprocessed_CT_scans = pickle.load(f)
with open(f'{args.data_dir}/y.pickle', 'rb') as f:
    labels = pickle.load(f)

X = np.array(preprocessed_CT_scans).astype(np.float32)
y = np.array(labels).astype(int).squeeze()

device = args.device if torch.cuda.is_available() else "cpu"

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

#setting path for model state_dict
state_dict_path = f'imagewise_models_state_dict/{args.model_dir}/'
if not os.path.exists(state_dict_path):
        os.makedirs(state_dict_path)

#setting path for saving model
model_path = f'imagewise_models/{args.model_dir}/'
if not os.path.exists(model_path):
        os.makedirs(model_path)

if args.tensorboard == True:
    #tensorboard log path
    tensorboard_path = f'imagewise_tensorboard/{args.model_dir}/'
    if not os.path.exists(tensorboard_path):
            os.makedirs(tensorboard_path)

    #writing log for tensorboard
    writer = SummaryWriter(log_dir = f'{tensorboard_path}/{args.model_name}')
torch.cuda.empty_cache()

#define pretrained model to use
model_name = args.model_name
try:
    if model_name in ['resnet18', 'resnet50','resnet152','vit_b_16','resnext50_32x4d','resnext101_32x8d',
                      'efficientnet_v2_s', 'efficientnet_v2_m', 'efficientnet_v2_l', 'vit_b_32', 'vit_l_16', 'swin_v2_t', 'swin_v2_s', 'swin_v2_b']:
        pass
    else:
        raise Exception

    if model_name == 'resnet18':
        model = models.resnet18(weights="DEFAULT")
    elif model_name == 'resnet50':
        model = models.resnet50(weights="DEFAULT")
    elif model_name == 'resnet152':
        model = models.resnet152(weights="DEFAULT")
    elif model_name == 'vit_b_16':
        model = models.vit_b_16(weights="DEFAULT")
    elif model_name == 'resnext50_32x4d':
        model = models.resnext50_32x4d(weights="DEFAULT")
    elif model_name == 'resnext101_32x8d':
        model = models.resnext101_32x8d(weights="DEFAULT")
    elif model_name == 'efficientnet_v2_s':
        model = models.efficientnet_v2_s(weights="DEFAULT")
    elif model_name == 'efficientnet_v2_m':
        model = models.efficientnet_v2_m(weights="DEFAULT")
    elif model_name == 'efficientnet_v2_l':
        model = models.efficientnet_v2_l(weights="DEFAULT")
    elif model_name == 'vit_b_32':
        model = models.vit_b_32(weights="DEFAULT")
    elif model_name == 'vit_l_16':
        model = models.vit_l_16(weights="DEFAULT")
    elif model_name == 'swin_v2_t':
        model = models.swin_v2_t(weights="DEFAULT")
    elif model_name == 'swin_v2_s':
        model = models.swin_v2_s(weights="DEFAULT")
    elif model_name == 'swin_v2_b':
        model = models.swin_v2_b(weights="DEFAULT")

except:
    print('Error: wrong model name')


# Freeze all the layers in the pre-trained model
for param in model.parameters():
    param.requires_grad = False

num_classes = 3

#redefine fc layer
if model_name.find('resnet')==0:
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features = in_features, out_features = num_classes)
    )
elif model_name.find('vit')==0:
    in_features = model.heads.head.in_features
    model.heads = nn.Sequential(
        nn.Linear(in_features=in_features, out_features=num_classes, bias=True)
        )
elif model_name.find('resnext')==0:
    model.fc = nn.Sequential(
        nn.Linear(in_features=2048, out_features=num_classes, bias=True)
    )
elif model_name.find('efficientnet')==0:
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=in_features, out_features=num_classes, bias=True)
    )
elif model_name.find('swin') == 0:
    in_features = model.head.in_features
    model.head = nn.Sequential(
        nn.Linear(in_features=in_features,
                  out_features=num_classes, bias=True)
    )

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()    
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=[args.beta1, args.beta2], eps=args.epsilon, weight_decay=args.weight_decay)

# Create instances of the custom dataset for train, test and validation set
train_dataset = ImgwiseDatset(X_train, y_train)
test_dataset = ImgwiseDatset(X_test, y_test)
val_dataset = ImgwiseDatset(X_val, y_val)

# Define the batch size 
batch_size = args.batch_size 

#dataloader
train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size, shuffle = False)
val_loader = DataLoader(val_dataset, batch_size, shuffle = False)

# Train the model
model.to(device)
num_epochs = args.epochs

#set initial values for earlystopping
best_model = None
best_val_loss = np.inf
patience = 20
patience_check = 0

for epoch in tqdm(range(num_epochs)):
    #train
    model.train()
    train_loss = 0.0
    train_acc = 0.0
    for i, (images, labels) in enumerate(train_loader):
        # Move the images and labels to the GPU if available
        images = images.to(device)
        labels = labels.to(device)
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        #loss
        train_loss += loss.item()*images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        train_acc += (predicted == labels).sum().item()
        
    # Print loss and accuracy for epoch
    train_loss /= len(train_loader.dataset)
    train_acc /= len(train_loader.dataset)
    
    if args.tensorboard == True:
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Acc/train", train_acc, epoch)   
    print('Epoch [{}/{}], Train Loss: {:.4f}, Train Accuracy: {:.4f}'.format(epoch+1, num_epochs, train_loss, train_acc))


    # validation
    
    model.eval()

    # Initialize variables to track loss and accuracy
    val_loss = 0.0
    val_acc = 0.0

    # Disable gradient computation
    with torch.no_grad():
        # Iterate over validation data
        for i, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Update loss and accuracy
            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            val_acc += (predicted == labels).sum().item()
        
        # Compute average loss and accuracy for epoch
        val_loss /= len(val_loader.dataset)
        val_acc /= len(val_loader.dataset)
        if args.tensorboard == True:
            writer.add_scalar("Loss/validation", val_loss, epoch)
            writer.add_scalar("Acc/validation", val_acc, epoch)
        print('Epoch [{}/{}], Validation Loss: {:.4f}, Validation Accuracy: {:.4f}'.format(epoch+1, num_epochs, val_loss, val_acc))
        
        #early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
            patience_check = 0

            #save model state_dict
            torch.save(best_model.state_dict(),f'{state_dict_path}{model_name}.ckpt')
            #save entire model
            torch.save(best_model, f'{model_path}{model_name}.ckpt')
            
        else:
            patience_check += 1 

        if patience_check >= patience:
            break

#Test
model.load_state_dict(torch.load(f'{state_dict_path}{model_name}.ckpt'))

model.eval()

# Initialize variables to track loss and accuracy
test_loss = 0.0
test_acc = 0.0
predicted_labels = []
true_labels = []

# Disable gradient computation
with torch.no_grad():
    # Iterate over validation data
    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Update loss and accuracy
        test_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        test_acc += (predicted == labels).sum().item()

        #store predicted and true labels
        predicted_labels.extend(predicted.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

# Compute average loss and accuracy for epoch
test_loss /= len(test_loader.dataset)
test_acc /= len(test_loader.dataset)

predicted_labels = np.array(predicted_labels)
true_labels = np.array(true_labels)
if args.tensorboard == True:
    writer.add_scalar("Loss/test", test_loss)
    writer.add_scalar("Acc/test", test_acc)     
# Print loss and accuracy for epoch
print('Test Loss: {:.4f}, Test Accuracy: {:.4f}'.format( test_loss, test_acc))

#classification_report
print("Classification report")
print(classification_report(true_labels, predicted_labels))

# Calculate and print confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(true_labels, predicted_labels))

if args.tensorboard ==True:
    writer.flush()
    writer.close()