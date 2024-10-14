import os
import numpy as np
import pandas as pd
import anndata
import scanpy as sc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from mygene import MyGeneInfo


class BinaryCellClassifier:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data_dir = "data"
        self.output_dir = "model_output"
        self.results_csv = os.path.join(self.output_dir, "binary_classification_rf_results.csv")
        self.adata = None
        self.unique_cell_types = None

        # Ensure directories exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize results DataFrame
        self.results_df = pd.DataFrame(columns=["Cell Type", "Negative Sampling", "Accuracy", "F1 Score", "AUROC", "AUPRC", "N Samples"])

    def load_and_preprocess_data(self):
        print("Loading data...")
        cells_data = np.load(self.file_path, allow_pickle=True).item()

        umi_counts = cells_data["UMI"]
        cell_classes = cells_data["classes"]
        gene_ids = cells_data["gene_ids"]

        print(f"Data Shape: {umi_counts.shape}, Gene IDs: {len(gene_ids)}, Cell Classes: {len(cell_classes)}")

        # Verify Ensembl Gene IDs and map to gene symbols
        mg = MyGeneInfo()
        gene_ensembl_ids = gene_ids.tolist()
        query = mg.querymany(gene_ensembl_ids, scopes="ensembl.gene", fields="symbol", species="human", returnall=True)
        ensembl_to_symbol = {item["query"]: item.get("symbol", "") for item in query["out"]}

        valid_indices = [i for i, ensembl_id in enumerate(gene_ensembl_ids) if ensembl_to_symbol.get(ensembl_id, "")]
        umi_counts = umi_counts[:, valid_indices]
        gene_symbols = [ensembl_to_symbol[gene_ensembl_ids[i]] for i in valid_indices]

        # Create AnnData object
        self.adata = anndata.AnnData(X=umi_counts)
        self.adata.var["gene_symbols"] = gene_symbols
        self.adata.obs["label"] = cell_classes
        self.adata.obs["cell_type"] = cell_classes

        self.unique_cell_types = np.unique(self.adata.obs["cell_type"])
        print(f"Number of unique cell types: {len(self.unique_cell_types)}")


    def split_data(self, adata, random_state=42):
        """
        Preprocess the data by normalizing, log-transforming, scaling
        """
        print("Normalizing total counts per cell...")
        sc.pp.normalize_total(adata, target_sum=1e4)
        
        print("Log-transforming data...")
        sc.pp.log1p(adata)

        
        print("Scaling data...")
        sc.pp.scale(adata, max_value=10)
        
        return adata


    def dynamic_even_split(self, adata, cell_type, n_negatives):
        """ 
        Dynamically distribute negative samples as evenly as possible across other cell types.
        Adjusts the number of negative samples if there are not enough samples available in any class.
        Ensures the total number of negative samples collected equals exactly n_negatives.
        """
        other_cell_types = [ct for ct in self.unique_cell_types if ct != cell_type]
        negatives_needed = n_negatives
        negative_samples = []
        class_sample_counts = []

        # First pass: collect as many as possible evenly
        for other_type in other_cell_types:
            other_samples = adata[adata.obs['cell_type'] == other_type]
            available_samples = other_samples.shape[0]
            n_samples = min(negatives_needed // len(other_cell_types), available_samples)
            if n_samples > 0:
                sampled_indices = np.random.choice(available_samples, n_samples, replace=False)
                negative_samples.append(other_samples.X[sampled_indices])
                negatives_needed -= n_samples
            class_sample_counts.append((other_type, available_samples - n_samples))

        # Second pass: handle any remainder by borrowing from classes with surplus samples
        if negatives_needed > 0:
            for other_type, surplus_samples in sorted(class_sample_counts, key=lambda x: x[1], reverse=True):
                if surplus_samples > 0 and negatives_needed > 0:
                    other_samples = adata[adata.obs['cell_type'] == other_type]
                    n_samples = min(surplus_samples, negatives_needed)
                    sampled_indices = np.random.choice(other_samples.shape[0], n_samples, replace=False)
                    negative_samples.append(other_samples.X[sampled_indices])
                    negatives_needed -= n_samples

        # Ensure exactly n_negatives are collected
        assert negatives_needed == 0, "Failed to collect the required number of negative samples"
        X_negative = np.vstack(negative_samples) if negative_samples else np.empty((0, adata.n_vars))

        return X_negative


    def train_random_forest_binary(self, adata, cell_type, negative_sampling='even'):
        positive_samples = adata[adata.obs['cell_type'] == cell_type]
        X_positive = positive_samples.X
        y_positive = np.ones(X_positive.shape[0])

        all_other_samples = adata[adata.obs['cell_type'] != cell_type]
        if negative_sampling == 'even':
            # Use dynamic even split for negatives
            X_negative = self.dynamic_even_split(adata, cell_type, X_positive.shape[0])
        else:
            # Randomly sample negatives
            random_negative_samples = all_other_samples[np.random.choice(all_other_samples.shape[0], X_positive.shape[0], replace=False)]
            X_negative = random_negative_samples.X

        y_negative = np.zeros(X_negative.shape[0])

        X = np.vstack([X_positive, X_negative])
        y = np.concatenate([y_positive, y_negative])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        rf.fit(X_train, y_train)

        y_pred = rf.predict(X_test)
        y_proba = rf.predict_proba(X_test)[:, 1]

        # Compute evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auroc = roc_auc_score(y_test, y_proba)
        auprc = average_precision_score(y_test, y_proba)
        n_samples = len(y_test)

        print(f"Accuracy for cell type {cell_type} (negative_sampling={negative_sampling}): {accuracy}")

        # Save metrics to results DataFrame
        self.results_df = self.results_df._append({
            "Cell Type": cell_type,
            "Negative Sampling": negative_sampling,
            "Accuracy": accuracy,
            "F1 Score": f1,
            "AUROC": auroc,
            "AUPRC": auprc,
            "N Samples": n_samples
        }, ignore_index=True)

        # Plot feature importance
        feature_importance = rf.feature_importances_
        top_10_indices = np.argsort(feature_importance)[-10:]
        top_10_genes = self.adata.var['gene_symbols'][top_10_indices]
        top_10_importances = feature_importance[top_10_indices]

        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_10_importances, y=top_10_genes)
        plt.title(f"Top 10 feature importances for {cell_type} (sampling={negative_sampling})")
        plt.xlabel("Importance")
        plt.ylabel("Gene")

        # Remove invalid characters from cell_type for file name
        safe_cell_type = cell_type.replace("/", "_").replace("\\", "_")

        plot_filename = os.path.join(self.output_dir, f"feature_importance_{safe_cell_type}_{negative_sampling}.png")
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()

    def save_results_to_csv(self):
        """Save the classification results to a CSV file."""
        self.results_df.to_csv(self.results_csv, index=False)
        print(f"Results saved to {self.results_csv}")

    def run_all(self):
        self.load_and_preprocess_data()
        adata = self.split_data(self.adata)

        for cell_type in self.unique_cell_types:
            # Run with evenly distributed negatives
            self.train_random_forest_binary(adata, cell_type, negative_sampling='even')

            # Run with randomly sampled negatives
            self.train_random_forest_binary(adata, cell_type, negative_sampling='random')

        self.save_results_to_csv()

if __name__ == "__main__":
    file_path = "cells.npy"
    classifier = BinaryCellClassifier(file_path)
    classifier.run_all()
