# Plots of optimizer and activation function experiments
"""
The first few sections store the results into dataframe and define functions for the plotting.
The last few sections show the plots.
The plots across validation scores look like scribbles when viewed on a small screen.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

### Store results into pandas dataframe
# Generate empty pandas dataframes to hold results
optim_df_acc = pd.DataFrame()
optim_df_losses = pd.DataFrame()
af_df_acc = pd.DataFrame()
af_df_losses = pd.DataFrame()

# Generate list of strings for optim and af results
optim_list = ["Adam", "SGD", "RMSProp"]
# af_list = ["ReLU", "SELU", "ELU", "LReLU", "Sigmoid", "Softplus"]
af_list = ["ReLU", "SELU", "ELU", "LReLU", "Softplus"]

fontsize = 15

# Add optimizer accuracies to dataframe
for i in optim_list:
    with open(f"res18_{i}_accs.txt", "r") as f:
        stats = f.read()
    stats = stats.split("\n")[0:-1]
    train = [float(stats[i]) for i in range(0, len(stats), 2)]
    val = [float(stats[i]) for i in range(1, len(stats), 2)]
    optim_df_acc[f"{i}_train"] = train
    optim_df_acc[f"{i}_val"] = val

# Add optimizer losses to dataframe
for i in optim_list:
    with open(f"res18_{i}_losses.txt", "r") as f:
        stats = f.read()
    stats = stats.split("\n")[0:-1]
    train = [float(stats[i]) for i in range(0, len(stats), 2)]
    val = [float(stats[i]) for i in range(1, len(stats), 2)]
    optim_df_losses[f"{i}_train"] = train
    optim_df_losses[f"{i}_val"] = val

# Add activation function accuracies to dataframe
for i in af_list:
    with open(f"res18_{i}_accs.txt", "r") as f:
        stats = f.read()
    stats = stats.split("\n")[0:-1]
    train = [float(stats[i]) for i in range(0, len(stats), 2)]
    val = [float(stats[i]) for i in range(1, len(stats), 2)]
    af_df_acc[f"{i}_train"] = train
    af_df_acc[f"{i}_val"] = val

# Add activation function losses to dataframe
for i in af_list:
    with open(f"res18_{i}_losses.txt", "r") as f:
        stats = f.read()
    stats = stats.split("\n")[0:-1]
    train = [float(stats[i]) for i in range(0, len(stats), 2)]
    val = [float(stats[i]) for i in range(1, len(stats), 2)]
    af_df_losses[f"{i}_train"] = train
    af_df_losses[f"{i}_val"] = val


### Plotting functions
def plot_single_result(prop: str, values: str, df):
    """
    First argument: string of optimizer or activation function
    Second argument: string; accuracy or losses
    Third argument: corresponding dataframe (e.g. optim_df_acc)
    """
    # Grab the results
    results = df[[f"{prop}_train", f"{prop}_val"]]
    results = results.reset_index()

    # Plot the results
    plt.plot(results["index"], results[f"{prop}_train"])
    plt.plot(results["index"], results[f"{prop}_val"])
    plt.xlabel("Epochs")
    plt.ylabel(f"{values}")
    plt.title(f"ResNet18 with {prop}")
    plt.show()


def plot_optim_vals(values: str, df, ylab: str):
    # Grab the results
    results = df[["SGD_val", "Adam_val", "RMSProp_val"]]
    results = results.reset_index()

    # Plot the results
    plt.plot(results["index"], results["SGD_val"])
    plt.plot(results["index"], results["Adam_val"])
    plt.plot(results["index"], results["RMSProp_val"])
    plt.xlabel("Epochs", fontsize=fontsize)
    plt.ylabel(ylab, fontsize=fontsize)
    plt.title(f"{values} of different optimizers", fontsize=fontsize)
    if values.lower() == "accuracies":
        plt.yticks(np.arange(0, 1, 0.1), fontsize=15)
    plt.legend(["SGD", "Adam", "RMSProp"], loc="lower right", fontsize=fontsize-5)
    plt.show()


def plot_af_vals(values: str, df, ylab: str):
    # Grab the results
    results = df[["ReLU_val", "SELU_val", "ELU_val", "LReLU_val", "Softplus_val"]]
    # results = df[["ReLU_val", "SELU_val", "ELU_val", "LReLU_val", "Sigmoid_val", "Softplus_val"]]
    results = results.reset_index()

    # Plot the results
    plt.plot(results["index"], results["ReLU_val"])
    plt.plot(results["index"], results["SELU_val"])
    plt.plot(results["index"], results["ELU_val"])
    plt.plot(results["index"], results["LReLU_val"])

    plt.plot(results["index"], results["Softplus_val"])
    plt.xlabel("Epochs", fontsize=fontsize)
    plt.ylabel(ylab, fontsize=fontsize)
    plt.title(f"{values} of different activation functions", fontsize=fontsize)
    if values.lower() == "accuracies":
        plt.yticks(np.arange(0, 1, 0.1), fontsize=fontsize)
    plt.legend(af_list, loc="lower right", fontsize=fontsize-5)
    plt.show()


# ### Validation Scores of optimizer experiments
plot_optim_vals("Accuracies", optim_df_acc, "Average accuracy")
# plot_optim_vals("Cross Entropy Loss", optim_df_losses)

# plot_af_vals("losses", af_df_losses)
plot_af_vals("Accuracies", af_df_acc, "Average accuracy")
# plot_optim_vals("Acc", optim_df_acc)
# plot_single_result("SGD", "Acc", optim_df_acc)

# for i in af_list:
#     plot_single_result(i, "Acc", af_df_acc)
