def get_config(model_name, data_name=None):
    config = {
        'vocab_size': 1024,
        'd_model': 256,
        'n_layers': 4,
        'n_head': 4,
        'd_ff': 512,
        'block_size': 256,
        'learning_rate': 1e-3,
        'max_iters': 10000,
        'eval_interval': 500,
        'batch_size': 64,
        'device': 'cpu' # Default to cpu for PythonAnywhere
    }
    
    if model_name == "neon004":
        config['d_ff_wide'] = 1024

    if model_name == "neon006":
        config['d_latent'] = 128  # MLA latent dimension

    # --- 10M scale models (neon011-014) ---
    if model_name == "neon011":  # Narrow & Deep
        config.update({'d_model': 384, 'n_layers': 8, 'n_head': 6, 'd_ff': 768})
    elif model_name == "neon012":  # Wide & Medium
        config.update({'d_model': 512, 'n_layers': 6, 'n_head': 8, 'd_ff': 1024})
    elif model_name == "neon013":  # Balanced
        config.update({'d_model': 320, 'n_layers': 8, 'n_head': 8, 'd_ff': 640})
    elif model_name == "neon014":  # MLP-Heavy (4× expansion)
        config.update({'d_model': 384, 'n_layers': 6, 'n_head': 6, 'd_ff': 1536})

    # --- 8-layer deep intent models (neon023-024) ---
    if model_name == "neon023":  # 8-layer neon016, no layerdrop
        config.update({'n_layers': 8, 'layerdrop': 0.0})
    elif model_name == "neon024":  # 8-layer neon016, layerdrop=0.1
        config.update({'n_layers': 8, 'layerdrop': 0.1})

    # --- 3M fair comparison models (neon026-030): d_ff scaled up ---
    if model_name == "neon026":    # neon005 baseline scaled
        config.update({'d_ff': 598})
    elif model_name == "neon027":  # neon010 Gated SDPA scaled
        config.update({'d_ff': 592})
    elif model_name == "neon028":  # neon006 MLA scaled
        config.update({'d_ff': 640})
    elif model_name == "neon029":  # neon001 GPT-2 scaled
        config.update({'d_ff': 891})
    elif model_name == "neon030":  # neon002 RMSNorm scaled
        config.update({'d_ff': 896})
    
    if model_name.startswith('neon'):
        if model_name in ['neon055', 'neon056', 'neon057', 'neon059', 'neon060']:
            config['d_ff'] = 592
        
        # Frankenstein Configs
        if model_name in [f"neon{i:03d}" for i in range(61, 214)]:
            config['d_model'], config['n_layers'], config['n_head'] = 256, 4, 4
            if model_name == 'neon061': config['d_ff'] = 2736 # ~10M
            elif model_name == 'neon066': config['d_ff'] = 172 # match neon016
            elif model_name in ['neon070', 'neon073', 'neon074', 'neon075']: config['d_ff'] = 576
            elif model_name == 'neon071': config['d_ff'] = 640
            elif model_name == 'neon072': config['d_ff'] = 512
            elif model_name == 'neon076': config['d_model'], config['d_ff'] = 240, 480
            elif model_name == 'neon077': config['d_ff'] = 368
            elif model_name == 'neon078': config['d_ff'] = 500
            elif model_name == 'neon079': config['d_ff'] = 480
            elif model_name == 'neon080': config['d_ff'] = 384 # Match neon016 Width
            elif model_name == 'neon081': config['d_ff'] = 378 # Match neon016 Context
            elif model_name == 'neon082': config['d_ff'] = 416 # Match neon016 ResHydra
            elif model_name == 'neon083': config['d_ff'] = 378 # Match neon016 (k=9 Modulation)
            elif model_name == 'neon084': config['d_ff'] = 382 # Match neon016 (k=5 Dilated)
            elif model_name == 'neon085': config['d_ff'] = 381 # Match neon016 (Dual Scale)
            elif model_name == 'neon086': config['d_ff'] = 380 # Match neon016 (Res Context)
            elif model_name == 'neon087': config['d_ff'] = 368 # Match neon016 (Pyramidal)
            elif model_name == 'neon088': config['d_ff'] = 381 # Match neon016 (Competitive)
            elif model_name == 'neon089': config['d_ff'] = 378 # Match neon016 (Dense Pyramidal)
            elif model_name == 'neon090': config['d_ff'] = 381 # Match neon016 (Asymmetric)
            elif model_name == 'neon091': config['d_ff'] = 2050 # Match neon061 (9.72M)
            elif model_name == 'neon092': config['d_ff'] = 2049 # Match neon061 (9.72M)
            elif model_name == 'neon093': config['n_layers'], config['d_ff'] = 8, 1240 # Match neon061 (9.72M)
            elif model_name == 'neon094': config['d_ff'] = 2113 # Hydra MLP but Standard Attention
            elif model_name == 'neon095': config['d_ff'] = 381 # Progressive Hydra
            elif model_name == 'neon096': config['d_ff'] = 437 # Heterogeneous Hydra/SwiGLU
            elif model_name == 'neon097': config['d_ff'] = 379 # Triple-Scale Hydra
            elif model_name == 'neon098': config['d_ff'] = 379 # Dilated Multi-Scale
            elif model_name == 'neon099': config['d_ff'] = 381 # Residual Hydra
            elif model_name == 'neon100': config['d_ff'] = 508 # Pure Hydra (Conv-Only)
            elif model_name == 'neon101': config['d_ff'] = 437 # 2x SwiGLU, 2x Hydra
            elif model_name == 'neon102': config['d_ff'] = 437 # Sandwich
            elif model_name == 'neon103': config['d_ff'] = 437 # Inv Sandwich
            elif model_name == 'neon104': config['d_ff'] = 472 # 3x SwiGLU, 1x Hydra
            elif model_name == 'neon105': config['d_ff'] = 472 # 1x Hydra, 3x SwiGLU
            elif model_name == 'neon106': config['d_ff'] = 381 # Dual-Gate Pure
            elif model_name == 'neon107': config['d_ff'] = 505 # Massive Reach Pure (RF 65)
            elif model_name == 'neon108': config['d_ff'] = 509 # Pure k=9 Only
            elif model_name == 'neon109': config['d_ff'] = 503 # Pure k=20 Reach
            elif model_name == 'neon110': config['d_ff'] = 509 # Pure k=9 Swish
            elif model_name == 'neon111': config['d_ff'] = 210 # Space-Aware Matrix (k=5)
            elif model_name == 'neon112': config['d_ff'] = 648 # Wide MLP with Bottleneck Gate
            elif model_name == 'neon113': config['d_ff'] = 507 # Conv-Attention Pure Hydra
            elif model_name == 'neon114': config['d_ff'] = 510 # Sharp-Value Conv-Attention
            elif model_name == 'neon115': config['d_ff'] = 508 # Multi-Head Conv-Attention
            elif model_name == 'neon116': config['d_ff'] = 507 # Full Multi-Head Conv-Attention
            elif model_name == 'neon117': config['d_ff'] = 507 # Activated Multi-Head
            elif model_name == 'neon118': config['d_ff'] = 510 # L2Norm Multi-Head
            elif model_name == 'neon119': config['d_ff'] = 338 # Dynamic Soft-Gate Hydra
            elif model_name == 'neon120': config['d_ff'] = 338 # Activated Intent Dynamic
            elif model_name == 'neon121': config['d_ff'] = 510 # Context-Aware Intent
            elif model_name == 'neon122': config['d_ff'] = 510 # Zero-Centered Norm
            elif model_name == 'neon123': config['d_ff'] = 508 # Residual Gated
            elif model_name == 'neon124': config['d_ff'] = 600 # MQI (Multi-Query)
            elif model_name == 'neon125': config['d_ff'] = 560 # Bottleneck Intent
            elif model_name == 'neon126': config['d_ff'] = 510 # No MLP Conv
            elif model_name == 'neon127': config['d_ff'] = 505 # Biased Attn Conv
            elif model_name == 'neon128': config['d_ff'] = 682 # No Intent Ablation
            elif model_name == 'neon129': config['d_ff'] = 570 # Hyper-Synergy (Full+MQI+Bias)
            elif model_name == 'neon130': config['d_ff'] = 572 # Sharp-V Synergy
            elif model_name == 'neon131': config['d_ff'] = 570 # Qwen-NexT Synergy (Centered)
            elif model_name == 'neon132': config['d_ff'] = 557 # Spectral Hydra
            elif model_name == 'neon133': config['d_ff'] = 550 # Commander Head (R=8)
            elif model_name == 'neon134': config['d_ff'] = 571 # Grouped Mamba Hybrid
            elif model_name == 'neon135': config['d_ff'] = 405 # Holographic
            elif model_name == 'neon136': config['d_ff'] = 252 # MoE Hydra (2 Experts)
            elif model_name == 'neon137': config['d_ff'] = 558 # Hierarchical Context
            elif model_name == 'neon138': config['d_ff'] = 563 # Strategic Colossus
            elif model_name == 'neon139': config['d_ff'] = 571 # Sequential Kernel Expansion
            elif model_name == 'neon140': config['d_ff'] = 500 # Parallel Spectral Heads
            elif model_name == 'neon141': config['d_ff'] = 572 # Denoising Bottleneck
            elif model_name == 'neon142': config['d_ff'] = 466 # Global Hum Hydra
            elif model_name == 'neon143': config['d_ff'] = 678 # Silent Hydra (Attention-Free)
            elif model_name == 'neon144': config['d_ff'] = 572 # Sigmoid Bottleneck
            elif model_name == 'neon145': config['d_ff'] = 507 # Multi-Head Denoising
            elif model_name == 'neon146': config['d_ff'] = 336 # Multi-Head Global Hum
            elif model_name == 'neon147': config['d_ff'] = 507 # Multi-Head Sigmoid
            elif model_name == 'neon148': config['d_ff'] = 571 # Sharp-Q Search
            elif model_name == 'neon149': config['d_ff'] = 574 # Dilated Context
            elif model_name == 'neon150': config['d_ff'] = 575 # Intent Recurrence
            elif model_name == 'neon151': config['d_ff'] = 488 # Inception Value
            elif model_name == 'neon152': config['d_ff'] = 507 # MHI Sharp-Q
            elif model_name == 'neon153': config['d_ff'] = 507 # MHI Dilated
            elif model_name == 'neon154': config['d_ff'] = 510 # MHI Persistence
            elif model_name == 'neon155': config['d_ff'] = 423 # MHI Inception
            elif model_name == 'neon156': config['d_ff'] = 675 # Spectral Silent
            elif model_name == 'neon157': config['d_ff'] = 680 # Wide-Merge Silent
            elif model_name == 'neon158': config['d_ff'] = 683 # Dilated Silent
            elif model_name == 'neon159': config['d_ff'] = 679 # Clean-Room Silent
            elif model_name == 'neon160': config['d_ff'] = 654 # Hybrid Ghost
            elif model_name == 'neon161': config['n_layers'], config['d_ff'] = 8, 207 # Deep Silent
            elif model_name == 'neon162': config['n_layers'], config['d_ff'] = 8, 194 # Deep Hybrid
            elif model_name == 'neon163': config['n_layers'], config['d_ff'] = 8, 155 # Alternating Ghost
            elif model_name == 'neon164': config['n_layers'], config['d_ff'] = 8, 204 # Pyramidal Silent
            elif model_name == 'neon165': config['n_layers'], config['d_ff'] = 8, 207 # Res-Gated Silent
            elif model_name == 'neon166': config['n_layers'], config['d_ff'] = 8, 205 # Deep Spectral
            elif model_name == 'neon167': config['d_model'], config['d_ff'] = 272, 1072 # 5.0M Non-Embed Class
            elif model_name in [f"neon{i}" for i in range(168, 188)]: 
                config['d_model'], config['d_ff'] = 272, 1072 # 5.0M Study Class
            elif model_name == 'neon213':
                config['d_model'], config['n_head'], config['n_layers'], config['d_ff'] = 384, 6, 8, 1536
                k = 21 if data_name and 'k21' in data_name else 9
                config['conv_k'], config['mlp_k'] = k, k  # Growable (FineWeb)
            else: config['d_ff'] = 512
        
        return config
            
    return config
