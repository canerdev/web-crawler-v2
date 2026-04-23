import torch
import torch.nn.functional as F
import json
import pandas as pd

from src.config_loader import load_config
from src.dataset import load_and_pad_embeddings
from src.e2e_wrapper import E2EWrapper
from src.diffusion_model import ConditionalDiffusion

def get_user_history(user_id_str, inter_path, mapping, max_seq_len):
    """
    It extracts the user's past interactions from a CSV file and converts them to a 1-indexed format.
    """
    df = pd.read_csv(inter_path, sep='\t')
    user_data = df[df['user_id:token'].astype(str) == str(user_id_str)]
    
    if user_data.empty:
        return []
        
    user_data = user_data.sort_values('timestamp:float')
    history = []
    for tid in user_data['item_id:token'].values:
        if str(tid) in mapping:
            history.append(mapping[str(tid)] + 1) # 1-indexed (0 = padding)
            
    return history[-max_seq_len:] # Only take the last max_seq_len

def predict_for_user(raw_user_id, top_k=10):
    # ── 1. Config ve Device Settings ──────────────────────────────────────
    cfg = load_config()
    active_ds = cfg['active_dataset']
    ds_cfg    = cfg['datasets'][active_ds]
    paths     = ds_cfg['paths']
    mdl       = cfg['model']
    tr        = cfg['training']
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"A suggestion is being made for user '{raw_user_id}' on device [{device}] ...")

    # ── 2. Load Mappings ──────────────────────────────
    with open(paths['mappings']['user'], 'r') as f:   user_mapping = json.load(f)
    with open(paths['mappings']['source'], 'r') as f: source_mapping = json.load(f)
    with open(paths['mappings']['target'], 'r') as f: target_mapping = json.load(f)
    
    # Reverse mapping to convert indexes from the model to actual Item IDs:
    # The model is running as 1-indexed, using val+1 as the key
    inv_target_mapping = {val + 1: key for key, val in target_mapping.items()}

    if str(raw_user_id) not in user_mapping:
        raise ValueError(f"User '{raw_user_id}' not found in the mapping file!")
    
    model_user_id = user_mapping[str(raw_user_id)] # 0-indexed ID

    # ── 3. Prepare User History ───────────────────────────────────
    max_seq_len = tr['max_seq_len']
    
    source_seq = get_user_history(raw_user_id, paths['inters']['source_train'], source_mapping, max_seq_len)
    target_seq = get_user_history(raw_user_id, paths['inters']['target_train'], target_mapping, max_seq_len)
    
    # Padding
    source_seq = source_seq + [0] * (max_seq_len - len(source_seq))
    target_seq = target_seq + [0] * (max_seq_len - len(target_seq))
    
    # Masks (True = padding)
    source_mask = [x == 0 for x in source_seq]
    target_mask = [x == 0 for x in target_seq]

    # Batch size = 1
    t_user_id    = torch.tensor([model_user_id], dtype=torch.long).to(device)
    t_source_seq = torch.tensor([source_seq], dtype=torch.long).to(device)
    t_target_seq = torch.tensor([target_seq], dtype=torch.long).to(device)
    t_source_msk = torch.tensor([source_mask], dtype=torch.bool).to(device)
    t_target_msk = torch.tensor([target_mask], dtype=torch.bool).to(device)

    # ── 4. Load Model ───────────────────
    padded_user_embs, padded_source_embs, padded_target_embs = load_and_pad_embeddings(
        pt_file_path   = paths['embeddings'],
        source_emb_key = ds_cfg['source_emb_key'],
        target_emb_key = ds_cfg['target_emb_key']
    )
    
    embed_dim = ds_cfg['model']['embed_dim']
    
    e2e_model = E2EWrapper(
        padded_user_embs=padded_user_embs, padded_source_embs=padded_source_embs,
        padded_target_embs=padded_target_embs, embed_dim=embed_dim,
        num_heads=mdl['num_heads'], dropout=mdl['dropout'], use_source_stream=mdl['use_source_stream']
    ).to(device)

    diffusion_model = ConditionalDiffusion(
        steps=mdl['diffusion']['steps'], item_dim=embed_dim, cond_dim=embed_dim,
        dropout=mdl['dropout'], p_uncond=mdl['diffusion']['p_uncond']
    ).to(device)

    # Upload the best trained model
    checkpoint = torch.load(paths['checkpoints']['best_model'], map_location=device)
    e2e_model.load_state_dict(checkpoint['e2e_state_dict'])
    diffusion_model.load_state_dict(checkpoint['diffusion_state_dict'])
    
    e2e_model.eval()
    diffusion_model.eval()

    # ── 5. Inference Process ────────────────────────────────────
    with torch.no_grad():
        # A. Generate the Condition Vector
        if mdl['use_source_stream']:
            c_ud = e2e_model(
                user_ids=t_user_id, target_seq_ids=t_target_seq, target_mask=t_target_msk,
                source_seq_ids=t_source_seq, source_mask=t_source_msk
            )
        else:
            c_ud = e2e_model(user_ids=t_user_id, target_seq_ids=t_target_seq, target_mask=t_target_msk)

        # B. Normalize all target product embeddings.
        all_target_embs = F.normalize(e2e_model.target_embedding.weight, p=2, dim=1)

        # C. Get suggestions using the diffusion model.
        top_k_indices = diffusion_model.sample(
            condition=c_ud,
            target_domain_embs=all_target_embs,
            watched_ids=t_target_seq,
            w=cfg['validation']['cfg_w'],
            k=top_k
        )

    # ── 6. Results ─────────────────────────────────────
    recommended_model_indices = top_k_indices[0].cpu().numpy() 
    
    print("\n--- RECOMMENDATION RESULTS ---")
    recommended_raw_ids = []
    for idx in recommended_model_indices:
        raw_id = inv_target_mapping.get(idx, "Unknown item")
        recommended_raw_ids.append(raw_id)
        print(f"- Recommended Item ID: {raw_id} (Model Index: {idx})")
        
    return recommended_raw_ids

if __name__ == "__main__":
    cfg = load_config()
    
    target_user = str(cfg['inference']['target_user_id'])
    k_val = cfg['inference'].get('top_k', 10)
    
    if not target_user or target_user == "None":
        print("ERROR: Please enter a valid user ID in the 'target_user_id' field in the config.yaml file")
    else:
        predict_for_user(raw_user_id=target_user, top_k=k_val)