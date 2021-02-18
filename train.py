import torch
import torchvision
from torch import optim
from torch import nn
from torchvision import models
from torchvision import datasets
from torchvision import transforms

import os
import time
import copy
from typing import Callable


def initialize_model(model_name, num_classes):
    model = None
    input_size = 0

    if model_name == "resnet18":
        """ Resnet18
        """
        model = models.resnet18()
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnet34":
        """ Resnet34
        """
        model = models.resnet34()
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    return model, input_size


def train_model(model, dataloaders: dict, criterion: Callable, optimizer, num_epochs=15):
    """ The function handles both the training and validation of a given model.
    It trains for the specified number of epochs and runs a full validation step afterwards, while keeping track of the
    best performing model (validation accuracy).
    It returns the best performing model."""
    since = time.time()
    val_acc_history = []
    # copy model weights
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs-1}')
        print('-'*10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to('cuda')
                labels = labels.to('cuda')
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            # deep copy the model
            if phase == 'val':
                val_acc_history.append(epoch_acc)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


data_dir = f"Dataset{os.path.sep}Food"
model_name = "resnet18"
num_classes = 11
batch_size = 8  # this can be increased accordingly, check CPU/GPU load, for me it only consumed 20% GPU
num_epochs = 15

model, input_size = initialize_model(model_name, num_classes)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

train_folder = "train"
validation_folder = "val"   # for parameter tuning
evaluation_folder = "eval"   # for performance assessment, not yet used in current git commit

# more transformations can be added later on
data_transforms = {
    train_folder: transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
    ]),
    validation_folder: transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
    ])
}

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in [train_folder, validation_folder]}
# Create training and validation dataloaders
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                   shuffle=True, num_workers=4) for x in [train_folder, validation_folder]}

params_to_update = model.parameters()
optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

# train and evaluate
criterion = nn.CrossEntropyLoss()
model, hist = train_model(model, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs)

# just some random variables to allow playing around in the console
val_dat = iter(dataloaders_dict['val'])
images, labels = val_dat.next()
images = images.to('cuda')
labels = labels.to('cuda')
