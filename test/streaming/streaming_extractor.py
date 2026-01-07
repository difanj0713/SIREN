import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import torch
import numpy as np
from utils.model_hooks import Qwen3RepresentationExtractor

class StreamingRepresentationExtractor(Qwen3RepresentationExtractor):
    def extract_prefixes_batch(self, input_ids_list, attention_mask_list):
        self.residual_outputs = []
        self.mlp_outputs = []

        batch_size = len(input_ids_list)
        input_ids_tensor = torch.tensor(input_ids_list, dtype=torch.long).to(self.device)
        attention_mask_tensor = torch.tensor(attention_mask_list, dtype=torch.long).to(self.device)

        with torch.no_grad():
            _ = self.model(input_ids=input_ids_tensor, attention_mask=attention_mask_tensor)

        batch_representations = []
        for batch_idx in range(batch_size):
            representations = {}
            for layer_idx in range(self.num_layers):
                residual_tensor = self.residual_outputs[layer_idx]
                mlp_tensor = self.mlp_outputs[layer_idx]

                if residual_tensor.dim() == 2:
                    residual_tensor = residual_tensor.unsqueeze(0)
                if mlp_tensor.dim() == 2:
                    mlp_tensor = mlp_tensor.unsqueeze(0)

                if residual_tensor.shape[0] != batch_size:
                    if residual_tensor.shape[1] == batch_size:
                        residual_tensor = residual_tensor.transpose(0, 1)

                if mlp_tensor.shape[0] != batch_size:
                    if mlp_tensor.shape[1] == batch_size:
                        mlp_tensor = mlp_tensor.transpose(0, 1)

                residual = residual_tensor[batch_idx]
                mlp = mlp_tensor[batch_idx]

                mask = attention_mask_tensor[batch_idx]
                valid_len = mask.sum().item()

                residual_valid = residual[:valid_len]
                mlp_valid = mlp[:valid_len]

                layer_rep = {}
                if "residual_mean" in self.rep_types:
                    layer_rep["residual_mean"] = residual_valid.mean(dim=0).cpu().float().numpy()
                if "mlp_mean" in self.rep_types:
                    layer_rep["mlp_mean"] = mlp_valid.mean(dim=0).cpu().float().numpy()
                representations[layer_idx] = layer_rep
            batch_representations.append(representations)

        return batch_representations
