# STILL UNDER EDITING!!!!
# Project Structure Overview

This repository contains a modular machine learning project with a focus on dimensionality reduction, clustering, and model implementation from scratch and reference. Below is a breakdown of the directory structure and the purpose of each component.

## 📂 Root Directory

General project documentation, configuration, and visualization.

- gitignore               # Git tracking exclusions
- README.md               # Project overview and structure
- tasks.txt               # Task list or project to-dos
- plots.ipynb             # Visualization notebook


## 📂 dimensionality_reduction

Modules and notebooks focused on PCA and clustering workflows.

- init.py                             # Package initializer
- clustering_test.ipynb                   # Clustering evaluation notebook
- filter_data.py                          # Data filtering utility
- pca_dimension_reduction_clustering.ipynb# PCA + clustering pipeline
- pca_model.ipynb                         # PCA model exploration


## 📂 methods

Reference and scratch implementations of various ML algorithms.

- init.py                   # Package initializer
- decision-tree-reference.py   # Decision tree reference implementation
- decision-tree-scratch.ipynb  # Decision tree from scratch
- filter_data.py               # Shared data filtering logic
- m3-reference.py              # Additional reference model (M3)
- neural-network-reference.py  # Neural network reference code
- neural-network-scratch.py    # Neural network from scratch
- random_forest.ipynb          # Random Forest model notebook


## 📂 src

Raw datasets and project documentation.

__MACOSX                    # System-generated folder (can be ignored)
- init.py                 # Package initializer
- claims_test.csv             # Test dataset
- claims_train.csv            # Training dataset
- ML_Project_Proposal_2025.pdf# Project proposal document
- ssn-3164764.pdf             # Supporting document (e.g., report or paper)


---

## 🧠 Notes

- All models are implemented either from scratch or using reference code for educational and experimental purposes.
- Data filtering utilities are reused across modules for consistency.
- Visualizations and clustering results are stored in dedicated notebooks for clarity.
- Proposal and documentation files are included for context and reproducibility.

---

## Instructions

If you desire to see the whole methods, you go in the methods folder and look at the full code of each method. Otherwise if you only wish to see the results of each method, you can open the method_collection.ipynb notebook.

---