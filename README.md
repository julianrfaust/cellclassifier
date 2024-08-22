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
