import matplotlib.pyplot as plt
import numpy as np
import os


def retrieve_epoch_statistics(path, model_type, stat_type):
    with open(f"{path}{os.path.sep}{model_type}_epoch_{stat_type}.txt", "r") as f:
        stats = f.read()
    stats = stats.split("\n")[0:-1]
    train_stats = [float(stats[i]) for i in range(0, len(stats), 2)]
    val_stats = [float(stats[i]) for i in range(1, len(stats), 2)]
    return train_stats, val_stats


def single_graph(data, stat_type, legend, title):
    """ Plot for a single curve of a certain statistic (e.g. accuracy or loss) """

    figure = plt.figure(figsize=(16, 12))
    x_axis = list(range(len(data)))
    plt.plot(x_axis, data)
    if max(data) <= 1:  # accuracies are always <= 1, losses are arbitrary
        plt.ylim(0, 1)
    else:
        plt.ylim(0, max(data) * 1.1)
    plt.yticks(np.linspace(0, max(data), 10).round(2))
    plt.xlabel("Time (epochs)")
    plt.ylabel(stat_type)
    plt.legend(legend)
    plt.title(title)
    figure.savefig(f"{title}.pdf")
    plt.close()


def multiple_graphs(data, stat_type, ylab, title, legend):
    """ Plot for multiple curves of the SAME type of statistic (e.g. both curves are accuracies or losses) """

    figure = plt.figure(figsize=(16, 12))
    x_axis = list(range(len(data[0])))
    fontsize = 15
    for dat in data:
        plt.plot(x_axis, dat)

    if stat_type == "accuracy":  # accuracies are always <= 1, losses are arbitrary
        plt.ylim(0, 1)
    elif stat_type == "loss":
        plt.ylim(0, max(min(data)) * 1.1)
    else:
        raise TypeError("Wrong stat_type was given. Possible options: ['accuracy', 'loss'].")

    plt.yticks(np.linspace(0, max(min(data)) * 1.1, 10).round(2), fontsize=fontsize)
    plt.xlabel("Time (epochs)", fontsize=fontsize)
    plt.ylabel(ylab, fontsize=fontsize)
    plt.legend(legend, fontsize=fontsize)
    plt.title(title, fontsize=fontsize)
    figure.savefig(f"{title}.pdf")
    plt.close()


def read_and_plot_non_regularized(main_data_folder, train_type, augment=False):
    epoch_path = f"model performances 100 epochs{os.path.sep}{main_data_folder}{os.path.sep}no regularization"
    train_path = f"{epoch_path}{os.path.sep}{train_type}{os.path.sep}"

    if augment:
        scr_train_losses, scr_val_losses = retrieve_epoch_statistics(train_path, f"{train_type}",
                                                                     "losses_train_augmented_1500")
        scr_train_accs, scr_val_accs = retrieve_epoch_statistics(train_path, f"{train_type}",
                                                                 "accs_train_augmented_1500")
    else:
        scr_train_losses, scr_val_losses = retrieve_epoch_statistics(train_path, f"{train_type}", "losses_train")

        scr_train_accs, scr_val_accs = retrieve_epoch_statistics(train_path, f"{train_type}", "accs_train")
                                                                 
    multiple_graphs([scr_train_losses, scr_val_losses], "loss",
                    "Averaged Cross-Entropy loss",
                    f"ResNet18 {train_type} losses {main_data_folder}",
                    ["Train error", "Validation error"])
    multiple_graphs([scr_train_accs, scr_val_accs], "accuracy",
                    "Averaged accuracy",
                    f"ResNet18 {train_type} accuracies {main_data_folder}",
                    ["Train accuracy", "Validation accuracy"])


def read_and_plot_regularized(main_data_folder, train_type, reg_type, reg_val):
    epoch_path = f"model performances 100 epochs{os.path.sep}{main_data_folder}{os.path.sep}regularization"
    train_path = f"{epoch_path}{os.path.sep}{train_type}ed{os.path.sep}{reg_type}"

    scr_train_losses, scr_val_losses = retrieve_epoch_statistics(train_path, f"{train_type}_{reg_type}-{reg_val}",
                                                                 "losses_train_augmented_1500")
    scr_train_accs, scr_val_accs = retrieve_epoch_statistics(train_path, f"{train_type}_{reg_type}-{reg_val}",
                                                             "accs_train_augmented_1500")

    multiple_graphs([scr_train_losses, scr_val_losses], "loss",
                    "Averaged Cross-Entropy loss",
                    f"ResNet18 {train_type}ed losses {main_data_folder} with {reg_type} ({reg_val})",
                    ["Train error", "Validation error"])
    multiple_graphs([scr_train_accs, scr_val_accs], "accuracy",
                    "Averaged accuracy",
                    f"ResNet18 {train_type}ed accuracies {main_data_folder} with {reg_type} ({reg_val})",
                    ["Train accuracy", "Validation accuracy"])


# PLOTTING: ORIGINAL DATASET EXPERIMENTS - NO REGULARIZATION
read_and_plot_non_regularized("on_original_dataset", "scratch-trained", augment=False)
read_and_plot_non_regularized("on_original_dataset", "pre-trained", augment=False)

# PLOTTING: AUGMENTED_1500 DATASET EXPERIMENTS - NO REGULARIZATION
read_and_plot_non_regularized("on_augmented_1500_dataset", "scratch-trained", augment=True)
read_and_plot_non_regularized("on_augmented_1500_dataset", "pre-trained", augment=True)

# PLOTTING: AUGMENTED_1500 DATASET EXPERIMENTS - REGULARIZATION (DROPOUT (P=[0.1, 0.2, 0.3]))
read_and_plot_regularized("on_augmented_1500_dataset", "scratch-train", "dropout", "0.1")
read_and_plot_regularized("on_augmented_1500_dataset", "scratch-train", "dropout", "0.2")
read_and_plot_regularized("on_augmented_1500_dataset", "scratch-train", "dropout", "0.3")

# PLOTTING: AUGMENTED_1500 DATASET EXPERIMENTS - REGULARIZATION (WEIGHT_DECAY = [0.1, 0.01, 0.001])
read_and_plot_regularized("on_augmented_1500_dataset", "scratch-train", "weight_dec", "0.1")
read_and_plot_regularized("on_augmented_1500_dataset", "scratch-train", "weight_dec", "0.01")
read_and_plot_regularized("on_augmented_1500_dataset", "scratch-train", "weight_dec", "0.001")
