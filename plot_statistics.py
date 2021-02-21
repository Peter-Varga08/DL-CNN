import matplotlib.pyplot as plt
import numpy as np
import os

epoch_path = f"model performances 100 epochs{os.path.sep}on_original_dataset{os.path.sep}"
scratch_path = f"{epoch_path}{os.path.sep}scratch-trained"
pretrained_path = f"{epoch_path}{os.path.sep}pre-trained"


def retrieve_epoch_statistics(path, model_type, stat_type):
    with open(f"{path}{os.path.sep}{model_type}_epoch_{stat_type}_train.txt", "r") as f:
        stats = f.read()
    stats = stats.split("\n")[0:-1]
    train_stats = [float(stats[i]) for i in range(0, len(stats), 2)]
    val_stats = [float(stats[i]) for i in range(1, len(stats), 2)]
    return train_stats, val_stats


def plot_single_statistic(data, stat_type, legend, title):
    """ Plot for a single type of data (e.g. accuracy or loss) """

    figure = plt.figure(figsize=(16, 12))
    x_axis = list(range(len(data)))
    plt.plot(x_axis, data)
    if max(data) <= 1:  # accuracies are always <= 1, losses are arbitrary
        plt.ylim(0, 1)
    else:
        plt.ylim(0, max(data)*1.1)
    plt.yticks(np.linspace(0, max(data), 10).round(2))
    plt.xlabel("Time (epochs)")
    plt.ylabel(stat_type)
    plt.legend(legend)
    plt.title(title)
    figure.savefig(f"{title}.pdf")
    plt.close()


def plot_multiple_statistics(data, stat_type, ylab, title, legend):
    """ Plot for multiple curves of the SAME type of statistic (e.g. both curves are accuracies or losses) """

    figure = plt.figure(figsize=(16, 12))
    x_axis = list(range(len(data[0])))
    fontsize = 15
    for dat in data:
        plt.plot(x_axis, dat)

    if stat_type == "accuracy":  # accuracies are always <= 1, losses are arbitrary
        plt.ylim(0, 1)
    elif stat_type == "loss":
        plt.ylim(0, max(min(data))*1.1)
    else:
        raise TypeError("Wrong stat_type was given. Possible options: ['accuracy', 'loss'].")

    plt.yticks(np.linspace(0, max(min(data))*1.1, 10).round(2), fontsize=fontsize)
    plt.xlabel("Time (epochs)", fontsize=fontsize)
    plt.ylabel(ylab, fontsize=fontsize)
    plt.legend(legend, fontsize=fontsize)
    plt.title(title, fontsize=fontsize)
    figure.savefig(f"{title}.pdf")
    plt.close()


scratch_train_losses, scratch_val_losses = retrieve_epoch_statistics(scratch_path, "scratch-trained", "losses")
pretrained_train_losses, pretrained_val_losses = retrieve_epoch_statistics(pretrained_path, "pre-trained", "losses")
plot_multiple_statistics([scratch_train_losses, scratch_val_losses], "loss",
                         "Averaged Cross-Entropy loss", "ResNet18 scratch-trained losses on non-augmented dataset",
                         ["Train error", "Validation error"])
plot_multiple_statistics([pretrained_train_losses, pretrained_val_losses], "loss",
                         "Averaged Cross-Entropy loss", "ResNet18 pre-trained losses on non-augmented dataset",
                         ["Train error", "Validation error"])

scratch_train_accs, scratch_val_accs = retrieve_epoch_statistics(scratch_path, "scratch-trained", "accs")
pretrained_train_accs, pretrained_val_accs = retrieve_epoch_statistics(pretrained_path, "pre-trained", "accs")
plot_multiple_statistics([scratch_train_accs, scratch_val_accs], "accuracy",
                         "Averaged accuracy", "ResNet18 scratch-trained accuracies on non-augmented dataset",
                         ["Train accuracy", "Validation accuracy"])
plot_multiple_statistics([pretrained_train_accs, pretrained_val_accs], "accuracy",
                         "Averaged accuracy", "ResNet18 pre-trained accuracies on non-augmented dataset",
                         ["Train accuracy", "Validation accuracy"])




