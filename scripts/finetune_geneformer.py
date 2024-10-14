import numpy as np
import pandas as pd
from scipy import sparse
from mygene import MyGeneInfo
import anndata
from geneformer import TranscriptomeTokenizer, Classifier
from ray import tune
import os

# Step 1: Load the Data
file_path = 'cells.npy'  # Update the path if needed
print("Loading data...")
cells_data = np.load(file_path, allow_pickle=True).item()

umi_counts = cells_data['UMI']  # Sparse matrix of UMI counts (cells x genes)
cell_classes = cells_data['classes']  # Array of cell type labels
gene_ids = cells_data['gene_ids']  # Array of gene IDs (likely Ensembl IDs)

print("Data loaded successfully.")
print(f"UMI Counts Type: {type(umi_counts)}")
print(f"Cell Classes Type: {type(cell_classes)}")
print(f"Gene IDs Type: {type(gene_ids)}")
print(f"UMI Counts Shape: {umi_counts.shape}")
print(f"Number of Gene IDs: {len(gene_ids)}")
print(f"Number of Cell Classes: {len(cell_classes)}")

# Step 2: Verify Ensembl Gene IDs
mg = MyGeneInfo()
gene_ensembl_ids = gene_ids.tolist()

print("Verifying Ensembl Gene IDs...")
query = mg.querymany(
    gene_ensembl_ids,
    scopes='ensembl.gene',
    fields='symbol',
    species='human',
    returnall=True
)

print(f"Total input gene Ensembl IDs: {len(gene_ensembl_ids)}")
print(f"Number of query results: {len(query['out'])}")

ensembl_to_symbol = {}
for item in query['out']:
    ensembl_id = item['query']
    if 'symbol' in item:
        ensembl_to_symbol[ensembl_id] = item['symbol']
    else:
        ensembl_to_symbol[ensembl_id] = ''

# Update the Ensembl IDs and remove those without valid mappings
valid_indices = [
    i for i, ensembl_id in enumerate(gene_ensembl_ids)
    if ensembl_to_symbol.get(ensembl_id, '') != ''
]
missing = [
    i for i, ensembl_id in enumerate(gene_ensembl_ids)
    if ensembl_to_symbol.get(ensembl_id, '') == ''
]

print(f"Number of genes with missing symbols: {len(missing)}")
if len(missing) > 0:
    print("Removing genes with missing symbols...")
    umi_counts = umi_counts[:, valid_indices]
    gene_symbols = [ensembl_to_symbol[gene_ensembl_ids[i]] for i in valid_indices]
    gene_ensembl_ids = [gene_ensembl_ids[i] for i in valid_indices]
else:
    gene_symbols = [ensembl_to_symbol[ensembl_id] for ensembl_id in gene_ensembl_ids]

print(f"Number of genes after removal: {len(gene_ensembl_ids)}")

if len(gene_ensembl_ids) == 0:
    print("Error: No genes remain after removing entries with missing symbols. Please check input data or gene mapping.")
    exit(1)

# Step 3: Create AnnData Object
print("Creating AnnData object...")
adata = anndata.AnnData(X=umi_counts)
adata.var['gene_symbols'] = gene_symbols
adata.var['ensembl_id'] = gene_ensembl_ids
adata.obs['label'] = cell_classes
adata.obs['n_counts'] = umi_counts.sum(axis=1).A1  # Sum of counts per cell
adata.obs['filter_pass'] = 1  # All cells pass

print(f"AnnData object created successfully with shape: {adata.shape}")

# Verify the number of unique cell types
unique_cell_types = np.unique(cell_classes)
print(f"Number of unique cell types: {len(unique_cell_types)}")
print(f"Sample cell type labels: {unique_cell_types[:10]}")

# Map cell classes to proper labels if necessary
cell_type_labels = cell_classes

adata.obs['cell_type'] = cell_type_labels

# Verify the updated cell types
unique_cell_types = np.unique(adata.obs['cell_type'])
print(f"Number of unique cell types after mapping: {len(unique_cell_types)}")
print(f"Unique cell types: {unique_cell_types}")

# Save the AnnData object
data_dir = 'data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

adata_file = os.path.join(data_dir, 'my_data.h5ad')
adata.write(adata_file)
print(f"AnnData object saved to '{adata_file}'.")

# Step 4: Tokenize the Data
print("Tokenizing the data...")
custom_attr_name_dict = {"cell_type": "cell_type"}
import pickle
metrics_list = []
# Load your existing token_dictionary.pkl
with open('token_dictionary.pkl', 'rb') as f:
    token_dictionary = pickle.load(f)
# for k, v in token_dictionary.items():
#     print(k, v)
with open('ensembl_mapping_dict.pkl', 'rb') as f:
    toke_dictionary = pickle.load(f)
# for k, v in toke_dictionary.items():
#     print(k, v)
# Instantiate the tokenizer
tk = TranscriptomeTokenizer(
    custom_attr_name_dict=custom_attr_name_dict,
    nproc=1,
    model_input_size=2048,
    special_token=False,
    gene_median_file='gene_median_dictionary.pkl',
    token_dictionary_file='token_dictionary.pkl',
    gene_mapping_file='ensembl_mapping_dict.pkl'
)

print(f"Tokenizer Model Input Size: {tk.model_input_size}")
print(f"Tokenizer Special Token: {tk.special_token}")

# Tokenize the data
tk.tokenize_data(
    data_directory=data_dir,
    output_directory='tokenized_data',
    output_prefix='my_data',
    file_format='h5ad'
)
print("Data tokenization complete.")

# Step 5: Fine-Tune Geneformer
eps=3
print("Fine-tuning the Geneformer model...")
training_args = {
    "num_train_epochs": eps,
    "learning_rate": 5e-5,
    "per_device_train_batch_size": 8,
    "seed": 42,
}

# Update the cell_state_dict with actual cell types
cc = Classifier(
    classifier="cell",
    cell_state_dict={
        "state_key": "cell_type",
        "states": list(unique_cell_types)
    },
    training_args=training_args,
    freeze_layers=3,
    num_crossval_splits=1,
    forward_batch_size=32,
    nproc=1,
    token_dictionary_file='token_dictionary.pkl'
)

# Print Classifier Details
print(f"Classifier Training Args: {cc.training_args}")
print(f"Classifier Freeze Layers: {cc.freeze_layers}")

cc.prepare_data(
    input_data_file='tokenized_data/my_data.dataset',
    output_directory='fine_tuning_output',
    output_prefix='my_fine_tuning'
)
print("Data preparation for fine-tuning complete.")

# Train the model
cc.validate(
    model_directory='Geneformer/gf-6L-30M-i2048',
    prepared_input_data_file='fine_tuning_output/my_fine_tuning_labeled_train.dataset',
    id_class_dict_file='fine_tuning_output/my_fine_tuning_id_class_dict.pkl',
    output_directory='fine_tuning_output',
    output_prefix='my_fine_tuning',

)
print("Model training complete.")

# Evaluate the model
all_metrics_test = cc.evaluate_saved_model(
    model_directory='fine_tuning_output/my_fine_tuning_model',  # Adjust the path as needed
    id_class_dict_file='fine_tuning_output/my_fine_tuning_id_class_dict.pkl',
    test_data_file='fine_tuning_output/my_fine_tuning_labeled_test.dataset',
    output_directory='fine_tuning_output',
    output_prefix='my_fine_tuning',
)
print("Model evaluation complete.")
print("Evaluation Metrics:")
print(all_metrics_test)

metrics_list.append({
            "epochs": eps,
            "learning_rate": 5e-5,
            **all_metrics_test})
# Optionally, print the evaluation metrics
metrics_df = pd.DataFrame(metrics_list)
metrics_df.to_csv('fine_tuning_output/eval_metrics.csv', index=False)