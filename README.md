# Deep Learning - Project 1
Peter Varga, s4291093  
Alfid Hadiat, s2863685

## Introduction
This repository contains the code used to generate the results used in a report for the first project of the Deep Learning course 2020-2021. The goal of the project was to examine a ResNet18's performance on an image dataset with different parameters through experiments. The following sections explains how the experiments were conducted and results were generated. 

## Dataset
The image dataset used to examine the ResNet18 was ...

## Optimizer and Activation Function Experiments 
The optimizer and activation function experiments were run in similar fashion. That is, each experiment is conducted using a Python script that runs a PyTorch implementation of ResNet18. These scripts are held within their corresponding folders (e.g. optimizer experiments in "Optimizer Experiments). Within each folder is also a set of bash scripts built for RuG's Peregrine cluster. To successfully run these scripts, run the bash scripts in the Peregrine in the same directory as the corresponding Python script and folder containing the Food dataset. Each script will generate a .txt file containing the accuracy and cross-entropy loss per epoch.  
The reader may also view these results in a Jupyter Notebook found in the "optim_af_results" folder. The results of these experiments found in the report are also in this folder. The notebook contains the functions used to read and store the results in a Pandas dataframe as well as plot them. Figures of the results used in the results are also in the notebook.