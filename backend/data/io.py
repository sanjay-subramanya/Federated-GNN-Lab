import os
import logging
import pandas as pd
from pathlib import Path
from config.settings import Config
from utils.blob_utils import upload_file_to_blob, download_file_from_blob
from utils.logging_utils import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

def load_protein_data():
    """Load protein expression data from local CSV cache if available, else get it from blob and cache."""
    filepath = os.path.join(Config.parent_dir, 'raw_data/protein_data.csv')

    if not os.path.exists(filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        logger.info("Local protein data not found, downloading from blob...")
        blob_key = "saved_models/raw_data/protein_data.csv"  

        try:
            download_file_from_blob(blob_key, str(filepath))
        except Exception as e:
            logger.error(f"Failed to download protein data from blob: {e}, falling back to remote URL.")
            
            url = "https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-BRCA.protein.tsv.gz"
            protein_df = pd.read_csv(
                url,
                sep='\t',
                compression='gzip',
                index_col='peptide_target',
                low_memory=False
            )
            protein_df.to_csv(filepath, encoding='utf-8')
            upload_file_to_blob(blob_key, str(filepath))
            logger.info(f"Uploaded protein data to blob with key: {blob_key}")
    else:
        logger.info("Loading cached protein data...")
        protein_df = pd.read_csv(filepath, index_col=0, encoding='utf-8')
    
    logger.info(f"Loaded protein data shape: {protein_df.shape}")
    protein_df.index = protein_df.index.astype(str).str.strip()
    protein_df = protein_df.T

    return protein_df

def load_phenotype_data():
    """Load phenotype data from local CSV cache if available, else get it from blob and cache."""
    filepath = os.path.join(Config.parent_dir, 'raw_data/phenotype_data.csv')

    if not os.path.exists(filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        logger.info("Local phenotype data not found, downloading from blob...")
        blob_key = "raw_data/phenotype_data.csv"

        try:
            download_file_from_blob(blob_key, str(filepath))
        except Exception as e:
            logger.error(f"Failed to download phenotype data from blob: {e}, falling back to remote URL.")
            
            url = "https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-BRCA.clinical.tsv.gz"
            phen_df = pd.read_csv(url, sep='\t', index_col=0)
            phen_df.to_csv(filepath, index=True, encoding='utf-8')
            upload_file_to_blob(blob_key, str(filepath))
            logger.info(f"Uploaded phenotype data to blob with key: {blob_key}")

    else:
        logger.info("Loading cached phenotype data...")
        phen_df = pd.read_csv(filepath, index_col=0, encoding='utf-8')
        
    logger.info(f"Loaded phenotype data shape: {phen_df.shape}")
    phen_df.index = phen_df.index.astype(str).str.strip()

    return phen_df


# def load_protein_data():
#     """Load protein expression data from local CSV cache if available, else download and cache it."""
#     filepath = 'raw_data/protein_data.csv'

#     if not Path(os.path.join(Config.parent_dir, filepath)).exists():
#         os.makedirs(os.path.join(Config.parent_dir, 'raw_data'), exist_ok=True)
#         logger.info("Downloading protein data...")
#         url = "https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-BRCA.protein.tsv.gz"
#         protein_df = pd.read_csv(
#             url,
#             sep='\t',
#             compression='gzip',
#             index_col='peptide_target',
#             low_memory=False  # allow mixed types, avoid misparse
#             # DO NOT pass dtype=str here!
#         )
#         logger.info(f"Downloaded protein data shape: {protein_df.shape}")
        
#         protein_df.to_csv(os.path.join(Config.parent_dir, filepath), encoding='utf-8')
#         logger.info(f"Saved protein data to CSV at {filepath}")
#     else:
#         logger.info("Loading cached protein data...")
#         protein_df = pd.read_csv(os.path.join(Config.parent_dir, filepath), index_col=0, encoding='utf-8')
#         logger.info(f"Loaded cached protein data shape: {protein_df.shape}")
    
#     # Clean index and transpose
#     protein_df.index = protein_df.index.astype(str).str.strip()
#     protein_df = protein_df.T
    
#     return protein_df

# def load_phenotype_data():
#     """Load phenotype data from local TSV cache if available, else download and cache it."""
#     filepath = 'raw_data/phenotype_data.csv'
    
#     if not Path(os.path.join(Config.parent_dir, filepath)).exists():
#         os.makedirs(os.path.join(Config.parent_dir, 'tcga-brca/raw_data'), exist_ok=True)
#         logger.info("Downloading phenotype data...")
#         url = "https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-BRCA.clinical.tsv.gz"
#         phen_df = pd.read_csv(url, sep='\t', index_col=0)
#         logger.info(f"Original downloaded shape: {phen_df.shape}")
        
#         # Save with proper parameters to preserve all data
#         phen_df.to_csv(os.path.join(Config.parent_dir, filepath), index=True, encoding='utf-8')
#         logger.info(f"Saved phenotype data with shape: {phen_df.shape}")
        
#     else:
#         logger.info("Loading cached phenotype data...")
#         # Load with proper parameters
#         phen_df = pd.read_csv(os.path.join(Config.parent_dir, filepath), index_col=0, encoding='utf-8')
#         logger.info(f"Loaded cached phenotype data shape: {phen_df.shape}")

#     # Ensure index is string and clean
#     phen_df.index = phen_df.index.astype(str).str.strip()
    
#     return phen_df
