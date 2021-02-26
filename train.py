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


def initialize_model(model_name, num_classes, pretrained=False):
    model = None

    if model_name == "resnet18":
        """ Resnet18 """
        model = models.resnet18(pretrained=pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == "resnet34":
        """ Resnet34 """
        model = models.resnet34(pretrained=pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    return model


def initialize_optimizer(optim_name, params, lr=0.001, momentum=0.0,
                         weight_decay=0.0, betas=(0, 0), eps=0):
    if optim_name == "SGD":
        optimizer = optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optim_name == "Adam":
        optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
    else:
        raise ValueError("Invalid optimizer name was given.")
    return optimizer


def train_model(model, dataloaders: dict, criterion: Callable, optimizer, regulrz=None, num_epochs=15):
    """ The function handles both the training and validation of a given model.
    It trains for the specified number of epochs and runs a full validation step afterwards, while keeping track of the
    best performing model (validation accuracy).
    It returns the best performing model."""
    since = time.time()
    val_acc_history = []
    # copy model weights
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    epoch_losses = []
    epoch_accs = []
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in [train_folder, validation_folder]:
            if phase == train_folder:
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to('cuda')
                if regulrz:
                    regulrz(inputs)
                labels = labels.to('cuda')
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if in train
                with torch.set_grad_enabled(phase == train_folder):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    if phase == train_folder:
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            epoch_losses.append(epoch_loss)
            epoch_accs.append(epoch_acc)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            # deep copy the model
            if phase == validation_folder:
                val_acc_history.append(epoch_acc)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, [epoch_losses, epoch_accs]


# This function will recursively replace all relu module to selu module.
def replace_act_funct(model, current, new):
    for child_name, child in model.named_children():
        if isinstance(child, current):
            setattr(model, child_name, new)
        else:
            replace_act_funct(child, current, new)


data_dir = f"Dataset{os.path.sep}Food"
num_classes = 10
batch_size = 64  # this can be increased accordingly, check CPU/GPU load, for me it only consumed 20% GPU
num_epochs = 100
input_size = 224

# original dropout_list and weight_decay_list are not doable in one peregrine job due to too high runtime,
# thus change regularizations according to experiment setting
dropout_list = [nn.Dropout(p=x, inplace=True) for x in [0.1, 0.2, 0.3]]
weight_decay_list = [1e-3, 1e-2, 1e-1]
# dropout_list = [nn.Dropout(p=0.1, inplace=True)]
# weight_decay_list = [1e-3]

train_folder = "train"
validation_folder = "val"  # for parameter tuning
evaluation_folder = "eval"  # for performance assessment, not yet used in current git commit

# more transformations can be added later on
data_transforms = {
    train_folder: transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    validation_folder: transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
}

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in
                  [train_folder, validation_folder]}
# Create training and validation dataloaders
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                   shuffle=True, num_workers=4) for x in
                    [train_folder, validation_folder]}
criterion = nn.CrossEntropyLoss()
device = ('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    model_name = "resnet18"
    model = initialize_model(model_name, num_classes)
    model = model.to(device)
    params_to_update = model.parameters()
    optim_name = "SGD"
    dropout = False
    weight_decay = False

    if dropout:
        for drpt in dropout_list:
            optimizer = initialize_optimizer(optim_name, params_to_update)
            model, hist, epoch_info = train_model(model, dataloaders_dict, criterion,
                                                  optimizer, regulrz=drpt, num_epochs=num_epochs)
            with open(f"scratch-train_dropout-{drpt.p}_epoch_losses_{train_folder}.txt", "w") as file_obj:
                for item in epoch_info[0]:
                    file_obj.write(f"{item}\n")
            with open(f"scratch-train_dropout-{drpt.p}_epoch_accs_{train_folder}.txt", "w") as file_obj:
                for item in epoch_info[1]:
                    file_obj.write(f"{item}\n")
    elif weight_decay:
        for wght_dec in weight_decay_list:
            optimizer = initialize_optimizer(optim_name, params_to_update, weight_decay=wght_dec)
            model, hist, epoch_info = train_model(model, dataloaders_dict, criterion,
                                                  optimizer, regulrz=None, num_epochs=num_epochs)
            with open(f"scratch-train_weight_dec-{wght_dec}_epoch_losses_{train_folder}.txt", "w") as file_obj:
                for item in epoch_info[0]:
                    file_obj.write(f"{item}\n")
            with open(f"scratch-train_weight_dec-{wght_dec}_epoch_accs_{train_folder}.txt", "w") as file_obj:
                for item in epoch_info[1]:
                    file_obj.write(f"{item}\n")
    else:
        optimizer = initialize_optimizer(optim_name, params_to_update)
        model, hist, epoch_info = train_model(model, dataloaders_dict, criterion,
                                              optimizer, regulrz=None, num_epochs=num_epochs)
        with open(f"scratch-train_epoch_losses_{train_folder}.txt", "w") as file_obj:
            for item in epoch_info[0]:
                file_obj.write(f"{item}\n")
        with open(f"scratch-train_epoch_accs_{train_folder}.txt", "w") as file_obj:
            for item in epoch_info[1]:
                file_obj.write(f"{item}\n")
