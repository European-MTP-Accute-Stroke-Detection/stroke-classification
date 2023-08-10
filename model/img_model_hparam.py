"""
Author: kiduk kang

Description: This script conducts hyperparameter tuning for image level classifier
"""
import optuna
from optuna.trial import TrialState
import logging
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm


class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        X = np.transpose(x, (2, 0, 1))
        y = y.squeeze()
        return X, y


def set_model(model_name, num_class=3):
    #defining model
    try:
        if model_name in ['resnet18', 'resnet50', 'resnet152', 'vit_b_16', 'resnext50_32x4d', 'resnext101_32x8d',
                          'efficientnet_v2_s', 'efficientnet_v2_m', 'efficientnet_v2_l', 'vit_b_32', 'vit_l_16',
                          'swin_v2_t','swin_v2_s','swin_v2_b']:
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

    num_classes = num_class

    #redefine fc layer
    if model_name.find('resnet') == 0:
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=num_classes)
        )
    elif model_name.find('vit') == 0:
        in_features = model.heads.head.in_features
        model.heads = nn.Sequential(
            nn.Linear(in_features=in_features,
                      out_features=num_classes, bias=True)
        )
    elif model_name.find('resnext') == 0:
        model.fc = nn.Sequential(
            nn.Linear(in_features=2048, out_features=num_classes, bias=True)
        )
    elif model_name.find('efficientnet') == 0:
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=in_features,
                      out_features=num_classes, bias=True)
        )
    elif model_name.find('swin') == 0:
        in_features = model.head.in_features
        model.head = nn.Sequential(
            nn.Linear(in_features=in_features,
                      out_features=num_classes, bias=True)
        )

    return model


def load_data(X, y, batch_size=32):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42)
    # Create instances of the custom dataset for train and test sets
    train_dataset = CustomDataset(X_train, y_train)
    test_dataset = CustomDataset(X_test, y_test)
    val_dataset = CustomDataset(X_val, y_val)

    # Define the batch size and create data loaders for train and test sets
    batch_size = batch_size

    #dataloader
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)

    return train_loader, test_loader, val_loader


def objective(trial):
    #search space
    model_name = trial.suggest_categorical(
        'model_name', ['resnet152', 'vit_l_16', 'efficientnet_v2_l', 'swin_v2_b', 'resnext101_32x8d'])
    #batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    lr = trial.suggest_float('lr', 1e-6, 1e-3, log=True)
    #beta1 = trial.suggest_float('beta1', 0.9, 0.999)
    #beta2 = trial.suggest_float('beta2', 0.8, 0.99)
    #epsilon = trial.suggest_float('epsilon', 1e-9, 1e-6, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-9, 1e-6, log=True)

    # model
    model = set_model(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #load data
    train_loader, _, val_loader = load_data(X, y)

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_model = None
    best_val_loss = np.inf
    patience = 10
    patience_check = 0
    model.train()
    for epoch in tqdm(range(2000)):
        #train
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

        print('Epoch [{}/{}], Train Loss: {:.4f}, Train Accuracy: {:.4f}'.format(
            epoch+1, 2000, train_loss, train_acc))

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
            print('Epoch [{}/{}], Validation Loss: {:.4f}, Validation Accuracy: {:.4f}'.format(
                epoch+1, 2000, val_loss, val_acc))

            trial.report(val_loss, epoch)

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.TrialPruned()

        #early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
            patience_check = 0

        else:
            patience_check += 1

        if patience_check >= patience:
            break

    return best_val_loss

if __name__ == '__main__':
    with open('data/14k_preprocessed_excl_scale/X.pickle', 'rb') as f:
        preprocessed_CT_scans = pickle.load(f)
    with open('data/14k_preprocessed_excl_scale/y.pickle', 'rb') as f:
        labels = pickle.load(f)

    X = np.array(preprocessed_CT_scans).astype(np.float32)
    y = np.array(labels).astype(int).squeeze()

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = "preprocess_excl_scale" 
    storage_name = "sqlite:///{}.db".format(study_name)
                    
    study = optuna.create_study(direction='minimize', study_name=study_name, storage=storage_name, load_if_exists=True)
    study.optimize(objective, n_trials=100)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

