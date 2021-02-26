# --------------------------------------------
# Library Imports
# --------------------------------------------
import torch
import torchvision
import torchvision.transforms as transforms
from torch import optim, nn
from torchvision import models, datasets
import os, time, copy

# --------------------------------------------
# Preprocessing and Settings
# --------------------------------------------
# Use CUDA device if it is available
device = ('cuda' if torch.cuda.is_available() else 'cpu')

# Generate the dataset transformers for pytorch
data_transforms = {
    'training': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'validation': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}

# Load the dataset into loaders
# Make sure to set the image folders in the same directory
image_datasets = {x: datasets.ImageFolder(
    os.path.join('Dataset', x),
    data_transforms[x]) for x in ['training', 'validation']}

dataloaders_dict = {
    x: torch.utils.data.DataLoader(
        image_datasets[x], batch_size = 64, shuffle = True,
        num_workers = 4
    )
    for x in ['training', 'validation']
}

# --------------------------------------------
# Functions for training
# --------------------------------------------
def train_model(model, dataloaders, criterion, optimizer, num_epochs = 15):
  since = time.time()
  val_acc_history = []
  best_model_wts = copy.deepcopy(model.state_dict())
  best_acc = 0.0
  best_acc_epoch = 0

  # Keep a record of epoch criterion loss and validation accuracy
  # across the epochs in arrays.
  epoch_losses = []
  epoch_accs = []

  for epoch in range(num_epochs):
    print(f"Epoch {epoch}/{num_epochs - 1}")
    print("-" * 10)

    # Each epoch has a training and validation phase
    for phase in ["training", "validation"]:
      if phase == "training":
        model.train()
      else:
        model.eval()
      
      running_loss = 0.0
      running_corrects = 0

      # Iterate over the data
      for inputs, labels in dataloaders[phase]:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # track the history only if in the training phase
        with torch.set_grad_enabled(phase == "training"):
          # get model outputs and calculate loss
          outputs = model(inputs)
          loss = criterion(outputs, labels)
          _, preds = torch.max(outputs, 1)

          # backward propogate and optimize when in training phase
          if phase == "training":
            loss.backward()
            optimizer.step()
        
        # update the statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
      
      epoch_loss = running_loss / len(dataloaders[phase].dataset)
      epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

      # Place the loss and accuracy in the arrays
      epoch_losses.append(epoch_loss)
      epoch_accs.append(epoch_acc.item())

      print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

      # deep copy the model
      if phase == "validation" and epoch_acc > best_acc:
        best_acc = epoch_acc
        best_acc_epoch = epoch
        best_model_wts = copy.deepcopy(model.state_dict())
      if phase == "validation":
        val_acc_history.append(epoch_acc)
  
  time_elapsed = time.time() - since
  print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
  print('Best val Acc: {:4f}'.format(best_acc))

  # Return the best model weights and epoch values
  model.load_state_dict(best_model_wts)
  return model, [epoch_losses, epoch_accs], best_acc

# Initialize the ResNet18 pretrained model
res18_model = models.resnet18(pretrained = True)
num_ftrs = res18_model.fc.in_features
res18_model.fc = nn.Linear(num_ftrs, 10) # 10 is the number of classes
input_size = 224

# Optimizers
res18_sgd = optim.SGD(res18_model.parameters(), lr = 0.001)
res18_adam = optim.Adam(res18_model.parameters(), lr = 0.001)
res18_rmsprop = optim.RMSprop(res18_model.parameters(), lr = 0.001)

# Criterion
criterion = nn.CrossEntropyLoss()

# --------------------------------------------
# Code when file is run
# --------------------------------------------
if __name__ == "__main__":
  # Put model into the GPU
  res18_model = res18_model.to(device)

  # Train and evaluate using Adam
  res18_Adam_model, res18_Adam_epochs, res18_Adam_best_acc = train_model(res18_model, dataloaders_dict, criterion, res18_adam, 100)

  # Write the epoch losses and accuracy into .txt files
  with open("res18_Adam_best_acc.txt", "w") as f:
      f.write(f"{res18_Adam_best_acc}")

  with open("res18_Adam_losses.txt", "w") as f:
      for item in res18_Adam_epochs[0]:
          f.write(f"{item}\n")

  with open("res18_Adam_accs.txt", "w") as f:
      for item in res18_Adam_epochs[1]:
          f.write(f"{item}\n")