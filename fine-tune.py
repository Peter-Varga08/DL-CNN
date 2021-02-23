from train import *
import pickle


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


model_name = "resnet18"
model = initialize_model(model_name, num_classes, pretrained=True)
set_parameter_requires_grad(model, feature_extracting=True)
model.fc.requires_grad_()  # set last layer's param.required_grad = True to enable learning
model = model.to(device)
params_to_update = model.parameters()
optimizer = initialize_optimizer("SGD", params_to_update)

# train and evaluate
model, hist, epoch_info = train_model(model, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs)

with open(f"pre-trained_epoch_losses_{train_folder}.txt", "w") as file_obj:
    for item in epoch_info[0]:
        file_obj.write(f"{item}\n")
with open(f"pre-trained_epoch_accs_{train_folder}.txt", "w") as file_obj:
    for item in epoch_info[1]:
        file_obj.write(f"{item}\n")
