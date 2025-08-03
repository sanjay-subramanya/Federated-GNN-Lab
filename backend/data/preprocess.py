import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from data.io import load_protein_data, load_phenotype_data
from config.settings import Config

def prepare_filtered_data(protein_df=None, phen_df=None, exclude_columns=None):

    if protein_df is None:
        protein_df = load_protein_data()
    if phen_df is None:
        phen_df = load_phenotype_data()

    label_column = "vital_status.demographic"

    protein_df.index = protein_df.index.str.strip()
    phen_df.index = phen_df.index.str.strip()

    merged_df = protein_df.join(phen_df, how='inner')
    merged_df = merged_df[merged_df[label_column].isin(Config.label_map.keys())].copy()
    merged_df['target_label'] = merged_df[label_column].map(Config.label_map)

    filtered_df = merged_df.dropna(axis=1, how='all')
    
    default_exclude_columns = [
    'ajcc_pathologic_stage.diagnoses',
    'stage_label',
    'id',
    'case_id',
    'submitter_id'
    ]
    if exclude_columns is None:
        exclude_columns = default_exclude_columns

    protein_cols = [
        col for col in filtered_df.columns
        if col not in exclude_columns + ['target_label'] and filtered_df[col].dtype.kind in ['i', 'f']
    ]

    X = filtered_df[protein_cols].values
    y = filtered_df['target_label'].values

    return X, y, filtered_df, protein_cols

def preprocess_features(X, y, n_neighbors=5, scale=True, apply_smote=True, seed=42):
    imputer = KNNImputer(n_neighbors=n_neighbors)
    X = imputer.fit_transform(X)

    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    if apply_smote:
        smote = SMOTE(random_state=seed)
        X, y = smote.fit_resample(X, y)

    return X, y

def preprocess_data(protein_df, phen_df):
    X, y, _, _ = prepare_filtered_data(protein_df, phen_df, exclude_columns=['id', 'case_id'])
    X, y = preprocess_features(X, y)
    class_names = list(Config.label_map.keys())
    return X, y, class_names