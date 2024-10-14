# CellClassifier

## 1. Overview
This project explores machine learning techniques for classifying cell types based on gene expression data. It encompasses baseline multiclass models, transfer learning, and binary classification approaches to provide a comprehensive understanding of cellular identities.

## 2. Table of Contents
1. [Overview](#1-overview)
2. [Table of Contents](#2-table-of-contents)
3. [Installation](#3-installation)
4. [Usage](#4-usage)
5. [Models](#5-models)
   - [Multiclass Baseline Models](#51-multiclass-baseline-models)
   - [Multiclass Transfer Learning](#52-multiclass-transfer-learning)
   - [Binary Classification](#53-binary-classification)
6. [Results](#6-results)

## 3. Installation
To set up the `cellclassifier`, follow these instructions:

```bash
git clone https://github.com/julianrfaust/cellclassifier.git
cd cellclassifier
pip install numpy pandas scipy mygene anndata scanpy scikit-learn matplotlib seaborn tqdm
```

For Geneformer installation, do:

```bash
git clone https://huggingface.co/ctheodoris/Geneformer
cd Geneformer
pip install .
```

Relevant `.pkl` files (ensembl_mapping_dict.pkl, gene_median_dictionary.pkl, gene_name_id_dict.pkl, token_dictionary.pkl) were downloaded from https://huggingface.co/ctheodoris/Geneformer/tree/main and moved into the relevant subdirectories. Minimal changes were made within `classifier.py`, `collator_for_classification.py`, `evaluation_utils.py` to reference these files. Alternatively, a `.zip` of the entire working folder (1.3GB) from which all scripts were run, with these minimal changes, can be accessed at the following Drive link:

## 4. Usage
To run the classification models, execute:

```bash
python multiclass_baselines.py
```
This script produces `combined_results.csv`, `gbdt_results.csv`, `random_forest_results.csv` found in `model_output/results`.

```bash
python finetune_geneformer.py
```
The output from this script is recorded in `fine_tuning/` in `model_output/results`, intermediate outputs and the finetuned model to `fine_tuning_outputs` (available at the drive link).

```bash
python binary_rf.py
```
This script saves `binary_rf_results.csv` to `model_output` along with `.pngs` of the top 10 genes by MDI feature importance for each cell type against a background distribution of random/balanced negatives.

## 5. Models

### 5.1 Multiclass Baseline Models
**Random Forest (RF) and Gradient Boosted Decision Trees (GBDT)**

Description: Utilizes RF and GBDT to classify cells from `cells.npy` into the 11 possible types based on their gene expression data (~16,000 dimensional).

### 5.2 Multiclass Transfer Learning
**Finetuned Geneformer**

Description: Utilizes transfer learning by finetuning Geneformer, a context-aware, attention-based deep learning model pretrained on large-scale transcriptomic data, adapting it to classify cell types. The smallest version of Geneformer (6 layers, 30M pretraining data) was used and fine-tuned for 3 epochs without thorough hyperparameter tuning due to computational constraints.

### 5.3 Binary Classification
**Random Forest (RF)**

Description: Binary classification to examine the influence of specific genes in determining cell types through feature importance of Random Forest model. Breakdown of results by cell type, with different negative schemes is implemented.

## 6. Results
Results in `csv/txt` form can be found in `model_output/results`, while plots can be found in `model_output/plots`. The comparison of RF/GBDT multiclass is in `combined_results.csv`, the log of fine tuning `geneformer-6L` for 3 epochs in `fine_tuning_log.txt`, and the binary classification results for each cell type can be found in the file `binary_classification_rf_results.csv`. Despite only finetuning for 3 epochs with the smallest Geneformer model (6L, 30M), the transfer learning approach substantially outperformed the simple RF/GBDT classifiers. While certain forms of preprocessing such as taking highly variable genes and PCA were able to boost performance of the RF/GBDT slightly (results not shown here), I was not able to match the 80% accuracy/0.67 macro-F1 achieved by transfer learning.




# Instructions
# SC-interview - Cell Annotation with Single-Cell RNA-seq

## Downloading the data

The data is here https://drive.google.com/drive/folders/1rL_AFCkEdKRvqKnj6PGAPJW6Raa16yF-?usp=drive_link

## Description

Single-cell RNA-seq encompasses various experimental protocols designed to measure the activity (or expression) of a set of genes within a single cell at a specific point in time. The resulting data is a high-dimensional vector of integer values, where each dimension represents the expression level of a particular gene. Researchers leverage single-cell RNA-seq data to identify unique cell signatures and understand gene function and interaction.

The measurement process for single-cell RNA-seq is intricate, but here's a simplified overview. Thousands of identifiable nucleotide chains are introduced into a single cell, where they bind to specific RNA molecules. These captured RNAs are then collected and compared against the human genome to determine their gene of origin. The expression level of a gene is quantified by counting how often its RNA sequences are matched in the genome, indicating the gene's activity within the cell at the time of the experiment. These RNA counts are also known as UMI counts (Unique Molecular Identifier).

## Data

The script file `interview.py` shows you how to read the data contained in `cells.npy`. The following dependencies are required numpy, scanpy. We strongly recommend using scanpy that has been specifically developed for the processing of single-cell data.

## Task

The task in this exercise is to train a classifier to categorize cells by their cell type using single-cell RNA-seq data. You may choose any model and metric to evaluate your results. You may also downsample the dataset if you do not have enough computing power to train your model. 

## Interview Process

For the interview we will ask you to present your work succintly. We're not looking for the highest possible performances but we expect clean code and in-depth understanding of all the choices you've made to tackle this classification task. We will ask questions about what you have done **but also broad machine learning questions that may be out of the scope of the homework.**

## References

- https://scanpy.readthedocs.io/en/stable/
- https://en.wikipedia.org/wiki/Single-cell_sequencing
- https://en.wikipedia.org/wiki/Single-cell_sequencing#Transcriptome_sequencing_(scRNA-seq)

If you face any problems (for example, downloading the data) or need precisions, please ask us. 

Good luck ! 



