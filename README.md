# Pharmaceutical-Sales-Prediction-across-multiple-stores

***

Time-series forecasting of sales using machine learning


**Table of Contents**

- [Pharmaceutical-Sales-Prediction-across-multiple-stores](#Pharmaceutical-Sales-Prediction-across-multiple-stores)
  - [Overview](#overview)
  - [About](#about)
  - [Project Structure](#project-structure)
    - [.dvc](#.dvc)
    - [.github](#.github)
    - [data](#data)
    - [notebooks](#notebooks)
    - [scripts](#scripts)
    - [training](#training)
    - [tests](#tests)
    - [root folder](#root-folder)

***

## Overview

This repository is used for week 3 challenge of 10Academy. The instructions for this project can be found in the challenge document.


## About
The finance team at Rossman Pharmaceuticals wants to forecast sales in all their stores across several cities six weeks ahead of time. Managers in individual stores rely on their years of experience as well as their personal judgment to forecast sales.

The data team identified factors such as promotions, competition, school and state holidays, seasonality, and locality as necessary for predicting the sales across the various stores.

The task was to build and serve an end-to-end product that delivers this prediction to analysts in the finance team.



![Alt text](MLFlow.png?raw=true "ML Flow")



## Project Structure
The repository has a number of files including python scripts, jupyter notebooks, raw and cleaned data, and text files. Here is their structure with a brief explanation.


### .dvc
- Data Version Control configurations

### .github
- a configuration file for github actions and workflow
- `workflows/CML.yml` continous machine learning configuration

### data
- the folder where the raw, and cleaned datasets' csv files are stored

### notebooks
- `EDA.ipynb`: a jupyter notebook that Explanatory Data Analysis
- `random forest model.ipynb`: a jupyter notebook that performs random forest modeling for our dataset


### scripts
- Different python utility scripts that have different purposes.

### training
- `train.py`: Trigerred by CML(Continous Machine Learning) and send training report and status

### tests


### root folder
- `requirements.txt`: a text file lsiting the projet's dependancies
- `.gitignore`: a text file listing files and folders to be ignored
- `README.md`: Markdown text with a brief explanation of the project and the repository structure.


