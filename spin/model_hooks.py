import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Hook class to extract representations catering for Qwen-3 model architecture
class Qwen3RepresentationExtractor:
    def __init__(self, model_path, device="cuda", batch_size=16, rep_types=None):
        self.device = device
        self.batch_size = batch_size
        self.rep_types = rep_types if rep_types else ["residual_mean", "mlp_mean"]
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map={"": self.device},
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()

        self.num_layers = len(self.model.model.layers)
        self.residual_outputs = []
        self.mlp_outputs = []
        self.hooks = []

    def _residual_hook(self, layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0].detach()
            else:
                hidden_states = output.detach()

            if len(self.residual_outputs) <= layer_idx:
                self.residual_outputs.append(hidden_states)
            else:
                self.residual_outputs[layer_idx] = hidden_states
        return hook

    def _mlp_hook(self, layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                mlp_output = output[0].detach()
            else:
                mlp_output = output.detach()

            if len(self.mlp_outputs) <= layer_idx:
                self.mlp_outputs.append(mlp_output)
            else:
                self.mlp_outputs[layer_idx] = mlp_output
        return hook

    def register_hooks(self):
        for idx, layer in enumerate(self.model.model.layers):
            residual_hook = layer.register_forward_hook(self._residual_hook(idx))
            mlp_hook = layer.mlp.register_forward_hook(self._mlp_hook(idx))
            self.hooks.append(residual_hook)
            self.hooks.append(mlp_hook)

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def extract_batch(self, texts):
        texts = [t.strip() if t.strip() else " " for t in texts]

        self.residual_outputs = []
        self.mlp_outputs = []

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            _ = self.model(**inputs)

        actual_batch_size = inputs["input_ids"].shape[0]
        batch_representations = []

        for batch_idx in range(actual_batch_size):
            representations = {}
            for layer_idx in range(self.num_layers):
                residual_tensor = self.residual_outputs[layer_idx]
                mlp_tensor = self.mlp_outputs[layer_idx]

                if residual_tensor.dim() == 2:
                    residual_tensor = residual_tensor.unsqueeze(0)
                if mlp_tensor.dim() == 2:
                    mlp_tensor = mlp_tensor.unsqueeze(0)

                if residual_tensor.shape[0] != actual_batch_size:
                    if residual_tensor.shape[1] == actual_batch_size:
                        residual_tensor = residual_tensor.transpose(0, 1)

                if mlp_tensor.shape[0] != actual_batch_size:
                    if mlp_tensor.shape[1] == actual_batch_size:
                        mlp_tensor = mlp_tensor.transpose(0, 1)

                residual = residual_tensor[batch_idx]
                mlp = mlp_tensor[batch_idx]

                mask = inputs["attention_mask"][batch_idx]
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

# Other model hook classes can be defined here as needed