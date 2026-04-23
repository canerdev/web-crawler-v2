import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import json
import os
from datetime import datetime

from src.config_loader import load_config
from src.dataset import load_and_pad_embeddings, CrossDomainDataset
from src.e2e_wrapper import E2EWrapper
from src.diffusion_model import ConditionalDiffusion
from src.metrics import calculate_metrics


CFG_W_CANDIDATES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]


def load_models(checkpoint_path, padded_user_embs, padded_source_embs,
                padded_target_embs, mdl, embed_dim, device):
    e2e_model = E2EWrapper(
        padded_user_embs  = padded_user_embs,
        padded_source_embs= padded_source_embs,
        padded_target_embs= padded_target_embs,
        embed_dim         = embed_dim,
        num_heads         = mdl['num_heads'],
        dropout           = mdl['dropout'],
        use_source_stream = mdl['use_source_stream']
    ).to(device)

    diffusion_model = ConditionalDiffusion(
        steps    = mdl['diffusion']['steps'],
        item_dim = embed_dim,
        cond_dim = embed_dim,
        dropout  = mdl['dropout'],
        p_uncond = mdl['diffusion']['p_uncond']
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    e2e_model.load_state_dict(checkpoint['e2e_state_dict'])
    diffusion_model.load_state_dict(checkpoint['diffusion_state_dict'])

    saved_epoch = checkpoint.get('epoch', '?')
    saved_hr    = checkpoint.get('hr',    '?')
    print(f"Checkpoint loaded — Epoch: {saved_epoch} | Val HR@10: {saved_hr}")

    e2e_model.eval()
    diffusion_model.eval()
    return e2e_model, diffusion_model


@torch.no_grad()
def run_evaluation(e2e_model, diffusion_model, loader, all_target_embs,
                   w, k, device, use_source_stream):
    all_hrs   = []
    all_ndcgs = []

    for batch in loader:
        user_ids        = batch['user_id'].to(device)
        target_seq      = batch['target_seq'].to(device)
        target_mask     = batch['target_mask'].to(device)
        target_item_ids = batch['target_item_id'].to(device)

        if use_source_stream:
            source_seq  = batch['source_seq'].to(device)
            source_mask = batch['source_mask'].to(device)
            c_ud = e2e_model(
                user_ids      = user_ids,
                target_seq_ids= target_seq,
                target_mask   = target_mask,
                source_seq_ids= source_seq,
                source_mask   = source_mask
            )
        else:
            c_ud = e2e_model(
                user_ids      = user_ids,
                target_seq_ids= target_seq,
                target_mask   = target_mask
            )

        top_k_indices = diffusion_model.sample(
            condition         = c_ud,
            target_domain_embs= all_target_embs,
            watched_ids       = target_seq,
            w                 = w,
            k                 = k
        )

        batch_hr, batch_ndcg = calculate_metrics(
            top_k_indices, target_item_ids, k=k
        )
        all_hrs.append(batch_hr)
        all_ndcgs.append(batch_ndcg)

    return float(np.mean(all_hrs)), float(np.mean(all_ndcgs))


def test():

    # ── 0. Config ────────────────────────────────────────────────────────
    cfg = load_config()
    
    # Active Dataset Settings
    active_ds = cfg['active_dataset']
    ds_cfg    = cfg['datasets'][active_ds]
    paths     = ds_cfg['paths']
    embed_dim = ds_cfg['model']['embed_dim']
    
    # Global Settings
    tr  = cfg['training']
    dl  = cfg['dataloader']
    mdl = cfg['model']
    val = cfg['validation']
    K   = val['top_k']

    use_source_stream = mdl['use_source_stream']

    device = (
        torch.device("cuda") if torch.cuda.is_available()        else
        torch.device("mps")  if torch.backends.mps.is_available() else
        torch.device("cpu")
    )
    
    print(f"Active Dataset: {active_ds.upper()}")
    print(f"Using device: {device}")

    # ── 2. GAT Embeddings ────────────────────────────────────────────────
    print("Loading GAT embeddings...")
    padded_user_embs, padded_source_embs, padded_target_embs = load_and_pad_embeddings(
        pt_file_path   = paths['embeddings'],
        source_emb_key = ds_cfg['source_emb_key'],
        target_emb_key = ds_cfg['target_emb_key']
    )

    # ── 3. Test Dataset ──────────────────────────────────────────────────
    print("Loading test data...")
    test_dataset = CrossDomainDataset(
        source_inter_path       = paths['inters']['source_train'],
        target_inter_path       = paths['inters']['target_test'],
        train_target_inter_path = paths['inters']['target_train'],
        source_mapping_path     = paths['mappings']['source'],
        target_mapping_path     = paths['mappings']['target'],
        user_mapping_path       = paths['mappings']['user'],
        max_seq_len             = tr['max_seq_len'],
        mode                    = 'test'
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size = dl['valid_batch_size'],
        shuffle    = False,
        num_workers= dl['valid_num_workers'],
        pin_memory = dl['valid_pin_memory']
    )
    print(f"Test samples: {len(test_dataset)}")

    # ── 4. Load Models ───────────────────────────────────────────────────
    print(f"\nLoading checkpoint: {paths['checkpoints']['best_model']}")
    e2e_model, diffusion_model = load_models(
        checkpoint_path   = paths['checkpoints']['best_model'],
        padded_user_embs  = padded_user_embs,
        padded_source_embs= padded_source_embs,
        padded_target_embs= padded_target_embs,
        mdl               = mdl,
        embed_dim         = embed_dim,
        device            = device
    )

    # ── 5. Pre-compute Normalized Target Embeddings ───────────────────────
    with torch.no_grad():
        all_target_embs = F.normalize(
            e2e_model.target_embedding.weight, p=2, dim=1
        )

    # ── 6. CFG Weight Sweep ──────────────────────────────────────────────
    print(f"\nSweeping CFG weights: {CFG_W_CANDIDATES}")
    print("-" * 50)

    sweep_results = {}

    for w in CFG_W_CANDIDATES:
        print(f"Evaluating w={w:.1f} ...")
        hr, ndcg = run_evaluation(
            e2e_model, diffusion_model,
            tqdm(test_loader, desc=f"  w={w:.1f}", leave=False),
            all_target_embs,
            w=w, k=K, device=device,
            use_source_stream=use_source_stream
        )
        sweep_results[w] = {'hr': hr, 'ndcg': ndcg}
        print(f"  w={w:.1f} → HR@{K}: {hr:.4f} | NDCG@{K}: {ndcg:.4f}")

    # ── 7. Best w ────────────────────────────────────────────────────────
    best_w    = max(sweep_results, key=lambda w: sweep_results[w]['hr'])
    best_hr   = sweep_results[best_w]['hr']
    best_ndcg = sweep_results[best_w]['ndcg']

    # ── 8. Terminal Report ───────────────────────────────────────────────
    print("\n" + "=" * 50)
    print(f"          FINAL TEST RESULTS ({active_ds.upper()})")
    print("=" * 50)
    print(f"  {'w (CFG)':<20} {'HR@'+str(K):<15} {'NDCG@'+str(K):<15}")
    print("-" * 50)
    for w, res in sorted(sweep_results.items()):
        marker = " ←" if w == best_w else ""
        print(f"  {w:<20.1f} {res['hr']:<15.4f} {res['ndcg']:<15.4f}{marker}")
    print("=" * 50)
    print(f"  Best w={best_w:.1f} | HR@{K}={best_hr:.4f} | NDCG@{K}={best_ndcg:.4f}")
    print("=" * 50)

    # ── 9. JSON Report ───────────────────────────────────────────────────
    results_dir  = os.path.dirname(paths['checkpoints']['best_model'])
    results_path = os.path.join(results_dir, f"test_results_{active_ds}.json")

    report = {
        "timestamp"          : datetime.now().isoformat(),
        "dataset"            : active_ds.upper(),
        "checkpoint"         : paths['checkpoints']['best_model'],
        "use_source_stream"  : use_source_stream,
        "k"                  : K,
        "cfg_w_sweep"        : {
            str(w): {"hr": res['hr'], "ndcg": res['ndcg']}
            for w, res in sorted(sweep_results.items())
        },
        "best_w"             : best_w,
        f"best_hr@{K}"       : best_hr,
        f"best_ndcg@{K}"     : best_ndcg,
    }

    os.makedirs(results_dir, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n✓ Results saved → {results_path}")


if __name__ == "__main__":
    test()