# GMM_bin

## Introduction
This repository provides a Python script for automated, multi-Gaussian binning of continuous datasets. By leveraging scikit-learn's Gaussian Mixture Models (GMMs), the tool evaluates individual features and dynamically selects the optimal number of bins using the Bayesian Information Criterion (BIC). Components are automatically sorted by their means, ensuring that output labels map logically from low to high values.


## Command-Line Usage Guide
GMM_bin is designed to be run directly from the terminal for rapid dataset processing. It accepts a continuous numeric CSV file and outputs a matching CSV containing the discrete, logically ordered bin labels.

### 1. Basic Execution
If you have a raw, unlabeled dataset (e.g., feature_matrix.csv) and want to use the default parameter search space (3 to 8 components), simply pass the data path:

```
python gmm_bin.py -data_path ./feature_matrix.csv
```

This will process the file and generate a binned_output.csv in your current directory.

### 2. Advanced Execution (Recommended)
For data that has a header row containing feature names or for specifying the algorithm's search space for finding the optimal Gaussian Mixture Model, utilize the code's input parameters.

Here is an example command utilizing all available parameters:

```
python gmm_bin.py -data_path ./labeled_feature_matrix.csv \
                  --labeled True \
                  --output ./binned_feature_matrix.csv \
                  --min_comp 2 \
                  --max_comp 5
```
