# Import required libraries
import os
import numpy as np
import pandas as pd
from scipy import sparse
from mygene import MyGeneInfo
import anndata
import scanpy as sc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    cohen_kappa_score,
)
# from geneformer import TranscriptomeTokenizer, Classifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from itertools import product
from tqdm import tqdm

warnings.filterwarnings("ignore")


class CellClassifierComparison:
    def __init__(self, file_path, n_splits=5):
        self.file_path = file_path
        self.data_dir = "data"
        self.tokenized_data_dir = "tokenized_data"
        self.output_dir = "model_output"
        self.adata = None
        self.unique_cell_types = None
        self.token_dictionary_file = "token_dictionary.pkl"
        self.gene_median_file = "gene_median_dictionary.pkl"
        self.gene_mapping_file = "ensembl_mapping_dict.pkl"
        self.n_splits = n_splits  # Number of data splits for error bars

        # Ensure directories exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.tokenized_data_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

    def load_and_preprocess_data(self):
        # Step 1: Load the Data
        print("Loading data...")
        cells_data = np.load(self.file_path, allow_pickle=True).item()

        umi_counts = cells_data["UMI"]  # Sparse matrix of UMI counts (cells x genes)
        cell_classes = cells_data["classes"]  # Array of cell type labels
        gene_ids = cells_data["gene_ids"]  # Array of gene IDs 
        print("Data loaded successfully.")
        print(f"UMI Counts Shape: {umi_counts.shape}")
        print(f"Number of Gene IDs: {len(gene_ids)}")
        print(f"Number of Cell Classes: {len(cell_classes)}")

        # Step 2: Verify Ensembl Gene IDs
        mg = MyGeneInfo()
        gene_ensembl_ids = gene_ids.tolist()

        print("Verifying Ensembl Gene IDs...")
        query = mg.querymany(
            gene_ensembl_ids,
            scopes="ensembl.gene",
            fields="symbol",
            species="human",
            returnall=True,
        )

        ensembl_to_symbol = {}
        for item in query["out"]:
            ensembl_id = item["query"]
            if "symbol" in item:
                ensembl_to_symbol[ensembl_id] = item["symbol"]
            else:
                ensembl_to_symbol[ensembl_id] = ""

        # Update the Ensembl IDs and remove those without valid mappings
        valid_indices = [
            i
            for i, ensembl_id in enumerate(gene_ensembl_ids)
            if ensembl_to_symbol.get(ensembl_id, "") != ""
        ]
        missing = [
            i
            for i, ensembl_id in enumerate(gene_ensembl_ids)
            if ensembl_to_symbol.get(ensembl_id, "") == ""
        ]

        print(f"Number of genes with missing symbols: {len(missing)}")
        if len(missing) > 0:
            print("Removing genes with missing symbols...")
            umi_counts = umi_counts[:, valid_indices]
            gene_symbols = [
                ensembl_to_symbol[gene_ensembl_ids[i]] for i in valid_indices
            ]
            gene_ensembl_ids = [gene_ensembl_ids[i] for i in valid_indices]
        else:
            gene_symbols = [
                ensembl_to_symbol[ensembl_id] for ensembl_id in gene_ensembl_ids
            ]

        print(f"Number of genes after removal: {len(gene_ensembl_ids)}")

        if len(gene_ensembl_ids) == 0:
            print(
                "Error: No genes remain after removing entries with missing symbols. Please check input data or gene mapping."
            )
            exit(1)

        # Step 3: Create AnnData Object
        print("Creating AnnData object...")
        self.adata = anndata.AnnData(X=umi_counts)
        self.adata.var["gene_symbols"] = gene_symbols
        self.adata.var["ensembl_id"] = gene_ensembl_ids
        self.adata.obs["label"] = cell_classes
        if sparse.issparse(umi_counts):
            self.adata.obs["n_counts"] = umi_counts.sum(axis=1).A1  # Sum per cell
        else:
            self.adata.obs["n_counts"] = umi_counts.sum(axis=1)  # Sum per cell
        self.adata.obs["filter_pass"] = 1  # All cells pass

        # Verify the number of unique cell types
        unique_cell_types = np.unique(cell_classes)
        print(f"Number of unique cell types: {len(unique_cell_types)}")

        # Map cell classes to proper labels if necessary
        cell_type_labels = cell_classes

        self.adata.obs["cell_type"] = cell_type_labels

        # Verify the updated cell types
        self.unique_cell_types = np.unique(self.adata.obs["cell_type"])
        print(
            f"Number of unique cell types after mapping: {len(self.unique_cell_types)}"
        )

    def random_split_within_cell_type(
        self, adata, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42
    ):
        train_adata_list = []
        val_adata_list = []
        test_adata_list = []

        for cell_type in adata.obs["cell_type"].unique():
            # Get indices of cells for the current cell type
            cell_type_indices = np.where(adata.obs["cell_type"] == cell_type)[0]

            # Split indices into train and temp (val + test)
            train_idx, temp_idx = train_test_split(
                cell_type_indices,
                test_size=(1 - train_ratio),
                random_state=random_state,
                shuffle=True,
            )

            # Calculate adjusted test size proportionally
            adjusted_test_size = test_ratio / (val_ratio + test_ratio)

            # Split temp indices into val and test
            val_idx, test_idx = train_test_split(
                temp_idx,
                test_size=adjusted_test_size,
                random_state=random_state,
                shuffle=True,
            )

            # Subset the AnnData object using the indices
            train_data = adata[train_idx].copy()
            val_data = adata[val_idx].copy()
            test_data = adata[test_idx].copy()

            train_adata_list.append(train_data)
            val_adata_list.append(val_data)
            test_adata_list.append(test_data)

        # Concatenate the splits, ensuring consistent genes across datasets
        train_adata = train_adata_list[0].concatenate(
            train_adata_list[1:], join="inner"
        )
        val_adata = val_adata_list[0].concatenate(val_adata_list[1:], join="inner")
        test_adata = test_adata_list[0].concatenate(test_adata_list[1:], join="inner")

        return train_adata, val_adata, test_adata

    def split_data(self, random_state=42):
        print("Splitting data into train, validation, and test sets within cell types...")
        adata = self.adata.copy()

        # Preprocessing steps
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.scale(adata, max_value=10)

        # Split the data
        train_adata, val_adata, test_adata = self.random_split_within_cell_type(
            adata, random_state=random_state
        )

        return train_adata, val_adata, test_adata



    def train_random_forest(self, train_adata, val_adata, test_adata):
        print("Training Random Forest Classifier with hyperparameter grid")
        results = []
        hyperparameters = {
            "n_estimators": [100],
            "max_depth": [None],
        }
        param_grid = list(
            product(hyperparameters["n_estimators"], hyperparameters["max_depth"])
        )

        # Prepare data
        X_train = train_adata.X
        y_train = train_adata.obs["cell_type"].values
        X_val = val_adata.X
        y_val = val_adata.obs["cell_type"].values
        X_test = test_adata.X
        y_test = test_adata.obs["cell_type"].values

        from sklearn.preprocessing import LabelEncoder

        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_val_enc = le.transform(y_val)
        y_test_enc = le.transform(y_test)

        best_val_accuracy = -np.inf
        best_metrics = None
        best_params = None
        best_model = None

        for n_estimators, max_depth in param_grid:
            print(f"Training with n_estimators={n_estimators}, max_depth={max_depth}")
            # Train Random Forest
            rf = RandomForestClassifier(
                n_estimators=n_estimators, max_depth=max_depth, random_state=42
            )
            rf.fit(X_train, y_train_enc)
            # Predictions
            y_train_pred = rf.predict(X_train)
            y_val_pred = rf.predict(X_val)
            y_test_pred = rf.predict(X_test)
            # Probabilities for log_loss
            y_train_prob = rf.predict_proba(X_train)
            y_val_prob = rf.predict_proba(X_val)
            y_test_prob = rf.predict_proba(X_test)

            # Compute metrics
            metrics = {}
            for (
                dataset_name,
                y_true_enc,
                y_pred_enc,
                y_prob,
            ) in [
                ("train", y_train_enc, y_train_pred, y_train_prob),
                ("val", y_val_enc, y_val_pred, y_val_prob),
                ("test", y_test_enc, y_test_pred, y_test_prob),
            ]:
                accuracy = accuracy_score(y_true_enc, y_pred_enc)
                balanced_accuracy = balanced_accuracy_score(y_true_enc, y_pred_enc)
                macro_f1 = f1_score(y_true_enc, y_pred_enc, average="macro")
                micro_f1 = f1_score(y_true_enc, y_pred_enc, average="micro")
                # Cross Entropy Loss
                ce_loss = log_loss(y_true_enc, y_prob)
                # Matthews Correlation Coefficient
                mcc = matthews_corrcoef(y_true_enc, y_pred_enc)
                # Cohen's Kappa
                kappa = cohen_kappa_score(y_true_enc, y_pred_enc)
                # Store metrics
                metrics[f"{dataset_name}_accuracy"] = accuracy
                metrics[f"{dataset_name}_balanced_accuracy"] = balanced_accuracy
                metrics[f"{dataset_name}_macro_f1"] = macro_f1
                metrics[f"{dataset_name}_micro_f1"] = micro_f1
                metrics[f"{dataset_name}_cross_entropy"] = ce_loss
                metrics[f"{dataset_name}_mcc"] = mcc
                metrics[f"{dataset_name}_kappa"] = kappa

            # Add hyperparameters
            metrics["method"] = "Random Forest"
            metrics["n_estimators"] = n_estimators
            metrics["max_depth"] = max_depth
            results.append(metrics)

            # Update best model based on validation accuracy
            if metrics["val_accuracy"] > best_val_accuracy:
                best_val_accuracy = metrics["val_accuracy"]
                best_metrics = metrics
                best_params = {"n_estimators": n_estimators, "max_depth": max_depth}
                best_model = rf

        # Save results as CSV
        results_df = pd.DataFrame(results)
        results_df.to_csv(
            os.path.join(self.output_dir, "random_forest_results.csv"), index=False
        )

        print(f"Best Random Forest params: {best_params}")
        return best_metrics, best_model

    def train_gbdt(self, train_adata, val_adata, test_adata):
        print(
            "Training Gradient Boosted Decision Trees (GBDT) with hyperparameter grid"
        )
        results = []
        hyperparameters = {"n_estimators": [10], "learning_rate": [0.1]}
        # Generate all combinations
        param_grid = list(
            product(hyperparameters["n_estimators"], hyperparameters["learning_rate"])
        )

        # Prepare data
        X_train = train_adata.X
        y_train = train_adata.obs["cell_type"].values
        X_val = val_adata.X
        y_val = val_adata.obs["cell_type"].values
        X_test = test_adata.X
        y_test = test_adata.obs["cell_type"].values

        from sklearn.preprocessing import LabelEncoder

        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_val_enc = le.transform(y_val)
        y_test_enc = le.transform(y_test)

        best_val_accuracy = -np.inf
        best_metrics = None
        best_params = None
        best_model = None

        for n_estimators, learning_rate in param_grid:
            print(
                f"Training with n_estimators={n_estimators}, learning_rate={learning_rate}"
            )
            # Train GBDT
            gbdt = GradientBoostingClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                random_state=42,
            )
            gbdt.fit(X_train, y_train_enc)
            # Predictions
            y_train_pred = gbdt.predict(X_train)
            y_val_pred = gbdt.predict(X_val)
            y_test_pred = gbdt.predict(X_test)
            # Probabilities for log_loss
            y_train_prob = gbdt.predict_proba(X_train)
            y_val_prob = gbdt.predict_proba(X_val)
            y_test_prob = gbdt.predict_proba(X_test)

            # Compute metrics
            metrics = {}
            for (
                dataset_name,
                y_true_enc,
                y_pred_enc,
                y_prob,
            ) in [
                ("train", y_train_enc, y_train_pred, y_train_prob),
                ("val", y_val_enc, y_val_pred, y_val_prob),
                ("test", y_test_enc, y_test_pred, y_test_prob),
            ]:
                accuracy = accuracy_score(y_true_enc, y_pred_enc)
                balanced_accuracy = balanced_accuracy_score(y_true_enc, y_pred_enc)
                macro_f1 = f1_score(y_true_enc, y_pred_enc, average="macro")
                micro_f1 = f1_score(y_true_enc, y_pred_enc, average="micro")
                # Cross Entropy Loss
                ce_loss = log_loss(y_true_enc, y_prob)
                # Matthews Correlation Coefficient
                mcc = matthews_corrcoef(y_true_enc, y_pred_enc)
                # Cohen's Kappa
                kappa = cohen_kappa_score(y_true_enc, y_pred_enc)
                # Store metrics
                metrics[f"{dataset_name}_accuracy"] = accuracy
                metrics[f"{dataset_name}_balanced_accuracy"] = balanced_accuracy
                metrics[f"{dataset_name}_macro_f1"] = macro_f1
                metrics[f"{dataset_name}_micro_f1"] = micro_f1
                metrics[f"{dataset_name}_cross_entropy"] = ce_loss
                metrics[f"{dataset_name}_mcc"] = mcc
                metrics[f"{dataset_name}_kappa"] = kappa

            # Add hyperparameters
            metrics["method"] = "GBDT"
            metrics["n_estimators"] = n_estimators
            metrics["learning_rate"] = learning_rate
            results.append(metrics)

            # Update best model based on validation accuracy
            if metrics["val_accuracy"] > best_val_accuracy:
                best_val_accuracy = metrics["val_accuracy"]
                best_metrics = metrics
                best_params = {
                    "n_estimators": n_estimators,
                    "learning_rate": learning_rate,
                }
                best_model = gbdt

        # Save results as CSV
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(self.output_dir, "gbdt_results.csv"), index=False)

        print(f"Best GBDT params: {best_params}")
        return best_metrics, best_model
    
    # def train_geneformer(self, train_adata, val_adata, test_adata):
    #     print("Fine-tuning Geneformer with hyperparameter grid...")
    #     results = []
    #     from itertools import product
    #     import pandas as pd

    #     # Updated hyperparameters to include freeze_layers
    #     hyperparameters = {
    #         "num_train_epochs": [5],
    #         "learning_rate": [1e-4, 1e-5],
    #         "freeze_layers": [0],
    #     }

    #     # Generate all combinations
    #     param_grid = list(
    #         product(
    #             hyperparameters["num_train_epochs"],
    #             hyperparameters["learning_rate"],
    #             hyperparameters["freeze_layers"],
    #         )
    #     )

    #     best_val_accuracy = -np.inf
    #     best_metrics = None
    #     best_params = None

    #     for num_train_epochs, learning_rate, freeze_layers in param_grid:
    #         print(
    #             f"Training Geneformer with num_train_epochs={num_train_epochs}, learning_rate={learning_rate}, freeze_layers={freeze_layers}"
    #         )
    #         training_args = {
    #             "num_train_epochs": num_train_epochs,
    #             "learning_rate": learning_rate,
    #             "per_device_train_batch_size": 12,  # Fixed batch size
    #             "seed": 42,
    #         }

    #         # Update the cell_state_dict with actual cell types
    #         cc = Classifier(
    #             classifier="cell",
    #             cell_state_dict={
    #                 "state_key": "cell_type",
    #                 "states": list(self.unique_cell_types),
    #             },
    #             training_args=training_args,
    #             freeze_layers=freeze_layers,  # Use freeze_layers from hyperparameters
    #             num_crossval_splits=1,
    #             forward_batch_size=200,
    #             nproc=1,  # Set to 1 to prevent multiprocessing issues
    #             ray_config=None,  # Not using ray_config for grid search
    #             token_dictionary_file=self.token_dictionary_file,
    #         )

    #         # Prepare data for training
    #         train_prefix = f"my_fine_tuning_train_epochs{num_train_epochs}_lr{learning_rate}_freeze{freeze_layers}"
    #         val_prefix = f"my_fine_tuning_val_epochs{num_train_epochs}_lr{learning_rate}_freeze{freeze_layers}"
    #         test_prefix = f"my_fine_tuning_test_epochs{num_train_epochs}_lr{learning_rate}_freeze{freeze_layers}"

    #         cc.prepare_data(
    #             input_data_file=os.path.join(
    #                 self.tokenized_data_dir, "my_data_train.dataset"
    #             ),
    #             output_directory=self.output_dir,
    #             output_prefix=train_prefix,
    #         )
    #         cc.prepare_data(
    #             input_data_file=os.path.join(
    #                 self.tokenized_data_dir, "my_data_val.dataset"
    #             ),
    #             output_directory=self.output_dir,
    #             output_prefix=val_prefix,
    #         )
    #         cc.prepare_data(
    #             input_data_file=os.path.join(
    #                 self.tokenized_data_dir, "my_data_test.dataset"
    #             ),
    #             output_directory=self.output_dir,
    #             output_prefix=test_prefix,
    #         )

    #         # Train the model
    #         cc.validate(
    #             model_directory="Geneformer/gf-6L-30M-i2048",
    #             prepared_input_data_file=os.path.join(
    #                 self.output_dir, f"{train_prefix}_labeled_train.dataset"
    #             ),
    #             id_class_dict_file=os.path.join(
    #                 self.output_dir, f"{train_prefix}_id_class_dictionary.pkl"
    #             ),
    #             output_directory=self.output_dir,
    #             output_prefix=f"geneformer_epochs{num_train_epochs}_lr{learning_rate}_freeze{freeze_layers}",
    #         )

    #         # Evaluate the model
    #         all_metrics_test = cc.evaluate_saved_model(
    #             model_directory=os.path.join(
    #                 self.output_dir,
    #                 f"geneformer_epochs{num_train_epochs}_lr{learning_rate}_freeze{freeze_layers}_model",
    #             ),
    #             id_class_dict_file=os.path.join(
    #                 self.output_dir, f"{train_prefix}_id_class_dictionary.pkl"
    #             ),
    #             test_data_file=os.path.join(
    #                 self.output_dir, f"{test_prefix}_labeled_train.dataset"
    #             ),
    #             output_directory=self.output_dir,
    #             output_prefix=f"geneformer_epochs{num_train_epochs}_lr{learning_rate}_freeze{freeze_layers}",
    #         )

    #         # Collect metrics
    #         metrics = {
    #             "method": "Geneformer",
    #             "num_train_epochs": num_train_epochs,
    #             "learning_rate": learning_rate,
    #             "freeze_layers": freeze_layers,
    #         }

    #         # Assuming all_metrics_test contains the required metrics
    #         for key in all_metrics_test:
    #             metrics[key] = all_metrics_test[key]

    #         results.append(metrics)

    #         # Update best model based on validation accuracy
    #         if metrics.get("val_accuracy", 0) > best_val_accuracy:
    #             best_val_accuracy = metrics["val_accuracy"]
    #             best_metrics = metrics
    #             best_params = {
    #                 "num_train_epochs": num_train_epochs,
    #                 "learning_rate": learning_rate,
    #                 "freeze_layers": freeze_layers,
    #             }

    #     # Save results as CSV
    #     results_df = pd.DataFrame(results)
    #     results_df.to_csv(
    #         os.path.join(self.output_dir, "geneformer_results.csv"), index=False
    #     )

    #     print(f"Best Geneformer params: {best_params}")
    #     return best_metrics


    # def tokenize_data(self):
    #     print("Tokenizing the data...")
    #     custom_attr_name_dict = {"cell_type": "cell_type"}
    #     # Ensure token dictionaries exist
    #     if (
    #         not os.path.exists(self.token_dictionary_file)
    #         or not os.path.exists(self.gene_median_file)
    #         or not os.path.exists(self.gene_mapping_file)
    #     ):
    #         raise FileNotFoundError("Token dictionary files not found.")

    #     # Instantiate the tokenizer
    #     tk = TranscriptomeTokenizer(
    #         custom_attr_name_dict=custom_attr_name_dict,
    #         nproc=1,  # Set to 1 to disable multiprocessing if needed
    #         model_input_size=2048,
    #         special_token=False,
    #         gene_median_file=self.gene_median_file,
    #         token_dictionary_file=self.token_dictionary_file,
    #         gene_mapping_file=self.gene_mapping_file,
    #     )

    #     # Tokenize the train, val, and test data
    #     tk.tokenize_data(
    #         data_directory=self.data_dir,
    #         output_directory=self.tokenized_data_dir,
    #         output_prefix="my_data_train",
    #         file_format="h5ad",
    #     )
    #     tk.tokenize_data(
    #         data_directory=self.data_dir,
    #         output_directory=self.tokenized_data_dir,
    #         output_prefix="my_data_val",
    #         file_format="h5ad",
    #     )
    #     tk.tokenize_data(
    #         data_directory=self.data_dir,
    #         output_directory=self.tokenized_data_dir,
    #         output_prefix="my_data_test",
    #         file_format="h5ad",
    #     )
    #     print("Data tokenization complete.")

    def plot_results(self, all_results):
        # all_results is a dictionary with keys: 'Random Forest', 'GBDT', 'Geneformer'
        # and values are lists of metrics dictionaries from each split

        metrics_to_plot = [
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "micro_f1",
            "cross_entropy",
            "mcc",
            "kappa",
        ]
        datasets = ["train", "val", "test"]

        # Prepare data for plotting
        plot_data = []
        for method, results_list in all_results.items():
            for metrics in results_list:
                for dataset in datasets:
                    data_point = {
                        "method": method,
                        "dataset": dataset,
                        "accuracy": metrics[f"{dataset}_accuracy"],
                        "balanced_accuracy": metrics[f"{dataset}_balanced_accuracy"],
                        "macro_f1": metrics[f"{dataset}_macro_f1"],
                        "micro_f1": metrics[f"{dataset}_micro_f1"],
                        "cross_entropy": metrics[f"{dataset}_cross_entropy"],
                        "mcc": metrics[f"{dataset}_mcc"],
                        "kappa": metrics[f"{dataset}_kappa"],
                    }
                    plot_data.append(data_point)

        df = pd.DataFrame(plot_data)

        df.to_csv(os.path.join(self.output_dir, "combined_results.csv"), index=False)

        for metric in metrics_to_plot:
            plt.figure(figsize=(10, 6))
            sns.barplot(
                data=df,
                x="dataset",
                y=metric,
                hue="method",
                ci="sd",
                capsize=0.1,
            )
            plt.ylabel(f"{metric.replace('_', ' ').title()}")
            plt.title(f"Model Comparison - {metric.replace('_', ' ').title()}")
            plt.tight_layout()
            plot_filename = os.path.join(
                self.output_dir, f"{metric}_comparison.png"
            )
            plt.savefig(plot_filename)
            plt.close()

    def run_all(self):
        self.load_and_preprocess_data()

        all_rf_results = []
        all_gbdt_results = []
        all_geneformer_results = []

        for split_num in range(self.n_splits):
            print(f"\n--- Split {split_num + 1}/{self.n_splits} ---")
            random_state = 42 + split_num  # Change random state for each split
            train_adata, val_adata, test_adata = self.split_data(random_state)

            # Save split datasets
            train_adata.write(os.path.join(self.data_dir, f"my_data_train_{split_num}.h5ad"))
            val_adata.write(os.path.join(self.data_dir, f"my_data_val_{split_num}.h5ad"))
            test_adata.write(os.path.join(self.data_dir, f"my_data_test_{split_num}.h5ad"))

            # Tokenize data for Geneformer
            # if split_num == 0:
            #     self.tokenize_data()

            # Train and evaluate models
            rf_metrics, _ = self.train_random_forest(train_adata, val_adata, test_adata)
            all_rf_results.append(rf_metrics)

            gbdt_metrics, _ = self.train_gbdt(train_adata, val_adata, test_adata)
            all_gbdt_results.append(gbdt_metrics)

            # geneformer_metrics = self.train_geneformer(train_adata, val_adata, test_adata)
            # all_geneformer_results.append(geneformer_metrics)

        all_results = {
            "Random Forest": all_rf_results,
            "GBDT": all_gbdt_results,
            # "Geneformer": all_geneformer_results,
        }


        self.plot_results(all_results)


if __name__ == "__main__":
    file_path = "cells.npy"
    classifier = CellClassifierComparison(file_path, n_splits=3)
    classifier.run_all()
