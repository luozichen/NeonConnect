import os
import sys
import threading
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import importlib
import numpy as np
from tokenizers import Tokenizer

# Use local config
from .config import get_config

class NeonModelEngine:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._vis_lock = threading.RLock()
        self.models_cache = {}
        self.tokenizers_cache = {}
        # Local paths relative to NeonConnect root
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.ckpt_dirs = [
            os.path.join(base_dir, "models")
        ]
        self.tok_dir = os.path.join(base_dir, "tokenizers")

    def get_available_models(self):
        ckpts = []
        seen_ids = set()
        for ckpt_dir in self.ckpt_dirs:
            if not os.path.isdir(ckpt_dir):
                continue
            for f in sorted(os.listdir(ckpt_dir)):
                if not f.endswith("_best.pth") or f in seen_ids:
                    continue
                seen_ids.add(f)
                stem = f.replace("_best.pth", "")
                parts = stem.split("_")
                model_name = parts[0]
                tok_name = parts[1] if len(parts) >= 2 else "tok1"
                data_name = "_".join(parts[2:]) if len(parts) >= 3 else "hp0"
                ckpts.append({
                    "id": f,
                    "model_name": model_name,
                    "tok_name": tok_name,
                    "data_name": data_name,
                    "label": f"{model_name} ({tok_name}/{data_name})"
                })
        return ckpts

    def find_tokenizer(self, tok_name, data_name):
        p = os.path.join(self.tok_dir, f"{data_name}_{tok_name}.json")
        if os.path.exists(p): return p
        data_alias = data_name.replace("hp0", "hp")
        p = os.path.join(self.tok_dir, f"{data_alias}_{tok_name}.json")
        if os.path.exists(p): return p
        p = os.path.join(self.tok_dir, f"{data_name}.json")
        if os.path.exists(p): return p
        if os.path.isdir(self.tok_dir):
            for f in os.listdir(self.tok_dir):
                if f.endswith(f"_{tok_name}.json") and (data_name in f or data_alias in f):
                    return os.path.join(self.tok_dir, f)
        return None

    def load_model(self, model_id):
        if model_id in self.models_cache:
            return self.models_cache[model_id]
        ckpt_path = None
        for d in self.ckpt_dirs:
            p = os.path.join(d, model_id)
            if os.path.exists(p):
                ckpt_path = p
                break
        if not ckpt_path:
            raise FileNotFoundError(f"Checkpoint {model_id} not found locally.")
        stem = model_id.replace("_best.pth", "")
        parts = stem.split("_")
        model_name = parts[0]
        tok_name = parts[1] if len(parts) >= 2 else "tok1"
        data_name = "_".join(parts[2:]) if len(parts) >= 3 else "hp0"
        tok_path = self.find_tokenizer(tok_name, data_name)
        if not tok_path:
            raise FileNotFoundError(f"Tokenizer not found for {model_id}")
        tokenizer = Tokenizer.from_file(tok_path)
        vocab_size = tokenizer.get_vocab_size()
        config = get_config(model_name)
        config['vocab_size'] = vocab_size
        cls_name = model_name.capitalize()
        mod_name = f"NeonConnect.networks.{model_name}"
        if mod_name not in sys.modules:
            mod = importlib.import_module(mod_name)
        else:
            mod = sys.modules[mod_name]
        ModelClass = getattr(mod, cls_name)
        model = ModelClass(config)
        state = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        model.load_state_dict(state)
        model.to(self.device)
        model.eval()
        self.models_cache[model_id] = (model, tokenizer, config)
        return model, tokenizer, config

    @torch.no_grad()
    def generate(self, model_id, prompt, max_new_tokens=100, temperature=1.0, top_k=50):
        model, tokenizer, config = self.load_model(model_id)
        encoded = tokenizer.encode(prompt)
        ids = encoded.ids
        idx = torch.tensor([ids], dtype=torch.long, device=self.device)
        block_size = config['block_size']
        generated = list(ids)
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
            generated.append(next_token.item())
            if next_token.item() == 0: break
        return tokenizer.decode(generated)

    def capture_visualization(self, model_id, prompt):
        # Global lock to prevent concurrent hook patching
        with self._vis_lock:
            # We don't need the double-sanitization wrapper anymore if we trust optimize_tensor_data
            # But the user said "remove some of your ultra-robust fixes".
            # I will rely on the robust optimize_tensor_data inside _capture_visualization_unsafe
            return self._capture_visualization_unsafe(model_id, prompt)

    def _capture_visualization_unsafe(self, model_id, prompt):
        model, tokenizer, config = self.load_model(model_id)
        encoded = tokenizer.encode(prompt)
        ids = encoded.ids
        tokens = encoded.tokens
        if len(ids) > config['block_size']:
            ids = ids[:config['block_size']]
            tokens = tokens[:config['block_size']]
        input_tensor = torch.tensor([ids], device=self.device)
        
        # New registry structure
        data_registry = {
            "layers": [{} for _ in range(config['n_layers'])], 
            "top_k_probs": None,
            "top_k_tokens": None
        }

        # 1. Hook F.scaled_dot_product_attention for Attention Matrix (Post-Softmax)
        real_sdpa = F.scaled_dot_product_attention
        sdpa_counter = [0] # Mutable counter
        def spy_sdpa(q, k, v, *args, **kwargs):
            # Capture attention weights
            L, S = q.size(-2), k.size(-2)
            scale = kwargs.get('scale', None)
            s = 1.0 / (q.size(-1) ** 0.5) if scale is None else scale
            logits = q @ k.transpose(-2, -1) * s
            
            # Causal mask
            mask = torch.triu(torch.ones(L, S, dtype=torch.bool, device=q.device), diagonal=1)
            logits.masked_fill_(mask, float("-inf"))
            
            w = torch.softmax(logits, dim=-1)
            
            layer_idx = sdpa_counter[0]
            if layer_idx < len(data_registry["layers"]):
                data_registry["layers"][layer_idx]["attn"] = w.detach().cpu()
                sdpa_counter[0] += 1
                
            return w @ v

        # 2. Module Hooks for Q, K, V, I (Raw and Conv)
        hooks = []
        
        def get_raw_hook(layer_idx, n_head, head_dim):
            def hook(module, inp, out):
                # out is [B, T, 4*D]
                B, T, _ = out.shape
                # Split
                splits = out.split(n_head * head_dim, dim=2)
                if len(splits) == 4:
                    q, k, v, i = splits
                    # Reshape to [B, H, T, D_head] for visualization consistency
                    # Note: neon167 implementation transposes to [B, T, H, D] first then [B, H, T, D] usually
                    # But here out is [B, T, D]. We want [B, H, T, D_head]? 
                    # Actually standard generic viz usually expects [B, H, T, D] or [H, T, D]
                    # Let's store as [H, T, D] (squeezing B=1 later)
                    
                    # shape: [B, T, n_head * head_dim] -> [B, T, n_head, head_dim] -> [B, n_head, T, head_dim]
                    q = q.view(B, T, n_head, head_dim).transpose(1, 2)
                    k = k.view(B, T, n_head, head_dim).transpose(1, 2)
                    v = v.view(B, T, n_head, head_dim).transpose(1, 2)
                    i = i.view(B, T, n_head, head_dim).transpose(1, 2)
                    
                    data_registry["layers"][layer_idx]["raw_q"] = q.detach().cpu()
                    data_registry["layers"][layer_idx]["raw_k"] = k.detach().cpu()
                    data_registry["layers"][layer_idx]["raw_v"] = v.detach().cpu()
                    data_registry["layers"][layer_idx]["raw_i"] = i.detach().cpu()
            return hook

        def get_conv_hook(layer_idx, name, n_head, head_dim):
            def hook(module, inp, out):
                # Conv1d out is [B, D, T]
                # We want [B, H, T, D_head]
                B, D, T = out.shape
                # Transpose to [B, T, D]
                x = out.transpose(1, 2)
                x = x.view(B, T, n_head, head_dim).transpose(1, 2)
                data_registry["layers"][layer_idx][name] = x.detach().cpu()
            return hook
            
        def get_mlp_hook(layer_idx):
             def hook(module, inp, out):
                 # inp[0] is x (input to mlp). out is result.
                 # Check neon167 mlp. It has gate.
                 # Capturing w2 input/output might be better?
                 # User said "Adding gating vector is okayish".
                 # Let's try to capture the gate if we can find it.
                 # neon167 MLP: forward(x): c9=... gate=sigmoid(...) return w2(gate*w1(x))
                 # Getting gate specifically requires hooking inside MLP or spying sigmoid.
                 # Let's spy sigmoid for MLP gates.
                 pass
             return hook

        # Spy sigmoid for MLP Gates (and Intent if needed, but we have Raw I)
        real_sigmoid = torch.sigmoid
        sigmoid_counter = [0]
        def spy_sigmoid(input):
            val = input.detach().cpu()
            res = real_sigmoid(input)
            # Heuristic: MLP gate is 3D [B, T, D] usually. Intent is 4D [B, H, T, D] inside attn?
            # In neon167:
            # Attn intent: `y = torch.sigmoid(intent) * attn_out`. Intent is [B, H, T, D] (transposed).
            # MLP gate: `gate = torch.sigmoid(...)`. Gate is [B, T, D_ff] or such? 
            # Wait, PureHydraMLP: c_gate_proj -> d_ff. So [B, T, d_ff].
            
            # Simple heuristic: if we are inside MLP (how do we know?)
            # Let's just store based on logic flow.
            # Block 0: Attn (sigmoid I) -> MLP (sigmoid G)
            
            # Actually, we have "raw_i" which IS the intent before sigmoid (and convolution).
            # The user wants "Gating vector".
            if val.dim() == 3: 
                # Likely MLP gate [B, T, D_ff]
                # We can try to assign to current layer. 
                # This depends on execution order: Attn then MLP.
                # sdpa_counter is incremented inside SDPA.
                # SDPA happens BEFORE MLP in the block.
                # So if sdpa_counter is K (meaning we just finished attn for layer K-1), we are in MLP for layer K-1?
                # No, sdpa_counter increments at END of SDPA.
                # Layer 0: SDPA (counter becomes 1) -> MLP.
                # So if counter is 1, we are in Layer 0 MLP.
                idx = sdpa_counter[0] - 1
                if 0 <= idx < len(data_registry["layers"]):
                    data_registry["layers"][idx]["mlp_gate"] = res.detach().cpu() # Capture OUTPUT of sigmoid
            return res

        for i, block in enumerate(model.blocks):
            # Raw QKVI
            hooks.append(block.attn.c_attn.register_forward_hook(get_raw_hook(i, config['n_head'], config['d_model'] // config['n_head'])))
            
            # Conv QKVI
            hooks.append(block.attn.conv_q.register_forward_hook(get_conv_hook(i, "conv_q", config['n_head'], config['d_model'] // config['n_head'])))
            hooks.append(block.attn.conv_k.register_forward_hook(get_conv_hook(i, "conv_k", config['n_head'], config['d_model'] // config['n_head'])))
            hooks.append(block.attn.conv_v.register_forward_hook(get_conv_hook(i, "conv_v", config['n_head'], config['d_model'] // config['n_head'])))
            hooks.append(block.attn.conv_i.register_forward_hook(get_conv_hook(i, "conv_i", config['n_head'], config['d_model'] // config['n_head'])))

        F.scaled_dot_product_attention = spy_sdpa
        torch.sigmoid = spy_sigmoid
        
        try:
            with torch.no_grad():
                logits, _ = model(input_tensor)
                last_logits = logits[0, -1].detach().cpu()
        finally:
            F.scaled_dot_product_attention = real_sdpa
            torch.sigmoid = real_sigmoid
            for h in hooks: h.remove()

        # Get Top-K tokens
        probs = torch.softmax(last_logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, 20)
        top_tokens = [tokenizer.decode([idx.item()]) for idx in top_indices]

        def optimize_tensor_data(t):
             """Recursively convert tensors to lists, rounding floats to 4 decimals."""
             if isinstance(t, torch.Tensor):
                 t = t.float().cpu().numpy()
             
             if isinstance(t, (np.ndarray, np.generic)):
                 # Strict sanitization
                 t = np.nan_to_num(t, nan=0.0, posinf=1e5, neginf=-1e5)
                 if isinstance(t, np.ndarray):
                    return np.round(t, 4).tolist()
                 else:
                    return float(np.round(t, 4))
             
             if isinstance(t, list): return [optimize_tensor_data(i) for i in t]
             if isinstance(t, dict): return {k: optimize_tensor_data(v) for k, v in t.items()}
             return t

        return {
            "tokens": tokens,
            # Flatten layers for easier frontend assumption or keep structured?
            # Old frontend expected keys like "attn", "q" which were lists of layers.
            # Let's reconstruct that format to minimize frontend rewrite, or just pass new format.
            # Passing new format "layers" is cleaner. I will update frontend.
            "layers": optimize_tensor_data(data_registry["layers"]),
            "top_k_probs": optimize_tensor_data(top_probs),
            "top_k_tokens": top_tokens,
            "config": config
        }
