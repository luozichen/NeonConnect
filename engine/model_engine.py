import os
import sys
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
        self.models_cache = {}
        self.tokenizers_cache = {}
        # Local paths relative to NeonConnect root
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.ckpt_dirs = [
            os.path.join(base_dir, "models")
        ]
        self.tok_dir = os.path.join(base_dir, "tokenizers")

    def get_available_models(self):
        """Scans the local models directory for available models."""
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
        model, tokenizer, config = self.load_model(model_id)
        encoded = tokenizer.encode(prompt)
        ids = encoded.ids
        tokens = encoded.tokens
        
        if len(ids) > config['block_size']:
            ids = ids[:config['block_size']]
            tokens = tokens[:config['block_size']]
            
        input_tensor = torch.tensor([ids], device=self.device)
        
        data_registry = {
            "attn": [],
            "q": [], "k": [], "v": [],
            "intent": [],
            "mlp": [],
            "conv": []
        }

        # SDPA Hook
        real_sdpa = F.scaled_dot_product_attention
        def spy_sdpa(q, k, v, *args, **kwargs):
            # q, k, v input to SDPA are usually [B, n_head, T, head_dim]
            data_registry["q"].append(q.detach().cpu())
            data_registry["k"].append(k.detach().cpu())
            data_registry["v"].append(v.detach().cpu())
            L, S = q.size(-2), k.size(-2)
            scale = kwargs.get('scale', None)
            s = 1.0 / (q.size(-1) ** 0.5) if scale is None else scale
            logits = q @ k.transpose(-2, -1) * s
            mask = torch.triu(torch.ones(L, S, dtype=torch.bool, device=q.device), diagonal=1)
            logits.masked_fill_(mask, float("-inf"))
            w = torch.softmax(logits, dim=-1)
            data_registry["attn"].append(w.detach().cpu())
            return w @ v

        # Intent Hook
        real_sigmoid = torch.sigmoid
        def spy_sigmoid(input):
            # Capture inputs to sigmoid. For Neon167 Attention it's the Intent vector.
            # Shape should be [B, n_head, T, head_dim] or [B, T, C]
            if input.dim() in [3, 4]:
                data_registry["intent"].append(input.detach().cpu())
            return real_sigmoid(input)

        # Conv Hook
        real_conv1d = F.conv1d
        def spy_conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
            out = real_conv1d(input, weight, bias, stride, padding, dilation, groups)
            # Capture depthwise intermediate states
            if groups > 1 and out.dim() == 3:
                data_registry["conv"].append({
                    "kernel": weight.shape[2],
                    "data": out.detach().cpu()
                })
            return out

        # MLP Hook
        mlp_hooks = []
        for block in model.blocks:
            target = getattr(block.mlp, 'w2', block.mlp)
            def make_hook():
                def hook(module, inp, out):
                    data_registry["mlp"].append({
                        "input": inp[0].detach().cpu(),
                        "output": out.detach().cpu()
                    })
                return hook
            mlp_hooks.append(target.register_forward_hook(make_hook()))

        F.scaled_dot_product_attention = spy_sdpa
        torch.sigmoid = spy_sigmoid
        F.conv1d = spy_conv1d
        
        try:
            with torch.no_grad():
                logits, _ = model(input_tensor)
                last_logits = logits[0, -1].detach().cpu()
        finally:
            F.scaled_dot_product_attention = real_sdpa
            torch.sigmoid = real_sigmoid
            F.conv1d = real_conv1d
            for h in mlp_hooks: h.remove()

        def to_list(t):
             if isinstance(t, torch.Tensor): return t.numpy().tolist()
             if isinstance(t, list): return [to_list(i) for i in t]
             if isinstance(t, dict): return {k: to_list(v) for k, v in t.items()}
             return t

        return {
            "tokens": tokens,
            "attn": to_list(data_registry["attn"]),
            "q": to_list(data_registry["q"]),
            "k": to_list(data_registry["k"]),
            "v": to_list(data_registry["v"]),
            "intent": to_list(data_registry["intent"]),
            "mlp": to_list(data_registry["mlp"]),
            "conv": to_list(data_registry["conv"]),
            "logits": to_list(last_logits),
            "config": config
        }
