import torch
import torch.nn.functional as F
import logging
from typing import Dict
from utils.logging_utils import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

def calculate_model_divergence(model1_state_dict: Dict[str, torch.Tensor], model2_state_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """
    Calculates the cosine divergence (1 - cosine similarity) between corresponding layers
    of two model state dictionaries.
    """
    divergence_scores = {}
    for (name1, param1) in model1_state_dict.items():
        if name1 in model2_state_dict:
            param2 = model2_state_dict[name1]
            # Ensure tensors are not scalar and have the same number of dimensions
            if param1.dim() > 0 and param2.dim() > 0 and param1.shape == param2.shape:
                # Flatten tensors for cosine similarity calculation
                v1 = param1.flatten()
                v2 = param2.flatten()

                # Handle cases where flattened vectors might be zero, causing NaN in cosine similarity
                if torch.norm(v1) == 0 or torch.norm(v2) == 0:
                    divergence_scores[name1] = 1.0 # Max divergence if one is zero vector
                else:
                    cosine_sim = F.cosine_similarity(v1, v2, dim=0)
                    divergence_scores[name1] = (1 - cosine_sim.item())
            else:
                logger.debug(f"Skipping divergence for layer {name1} due to non-tensor, scalar, or shape mismatch.")
        else:
            logger.debug(f"Layer {name1} in model1 not found in model2 for divergence calculation.")
    return divergence_scores