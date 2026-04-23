import torch
from torch.utils.data import Dataset
import json
import pandas as pd


def load_and_pad_embeddings(pt_file_path, source_emb_key, target_emb_key):
    """
    Loads GAT embeddings from the given path and prepends a zero PAD vector at index 0.

    Index conventions after padding:
      index 0    -> zero vector (padding token)
      index 1..N -> actual embeddings

    Downstream usage:
      user_ids          : 0-indexed in dataset -> E2EWrapper applies +1
      source/target seq : already +1 in dataset -> used directly
    """
    data       = torch.load(pt_file_path)
    embeddings = data['embeddings']

    keys = list(embeddings.keys())
    print(f"Embedding keys found: {keys}")

    if 'user' not in embeddings:
        raise KeyError(f"Missing 'user' key in embeddings. Found: {keys}")

    user_embs = embeddings['user']

    # Dynamically select source and target embeddings based on configuration
    if source_emb_key in embeddings and target_emb_key in embeddings:
        source_embs = embeddings[source_emb_key]
        target_embs = embeddings[target_emb_key]
    else:
        raise KeyError(
            f"Unexpected embedding keys: {keys}. "
            f"Expected '{source_emb_key}' and '{target_emb_key}' based on configuration."
        )

    embed_dim = user_embs.shape[1]
    print(f"embed_dim: {embed_dim}")
    print(f"Users: {user_embs.shape[0]} | Source({source_emb_key}): {source_embs.shape[0]} | Target({target_emb_key}): {target_embs.shape[0]}")

    pad = torch.zeros(1, embed_dim)
    return (
        torch.cat([pad, user_embs],   dim=0),  # padded_user_embs
        torch.cat([pad, source_embs], dim=0),  # padded_source_embs
        torch.cat([pad, target_embs], dim=0),  # padded_target_embs
    )


class CrossDomainDataset(Dataset):
    """
    Cross-domain sequential recommendation dataset.

    Source domain : Context signal (user's interaction history in source domain)
    Target domain : Predict next interaction in target domain

    Index conventions:
      user_id          : 0-indexed (raw mapping value)
      source_seq       : 1-indexed (0 = padding)
      target_seq       : 1-indexed (0 = padding)
      target_item_id   : 1-indexed

    Source history is time-bounded: only interactions BEFORE the target
    interaction's timestamp are included (prevents data leakage).
    """

    def __init__(
        self,
        source_inter_path,
        target_inter_path,
        source_mapping_path,
        target_mapping_path,
        user_mapping_path,
        max_seq_len=10,
        mode='train',
        train_target_inter_path=None 
    ):
        assert mode in ('train', 'valid', 'test'), \
            f"mode must be 'train', 'valid', or 'test', got '{mode}'"
        if mode != 'train':
            assert train_target_inter_path is not None, \
                "train_target_inter_path required for valid/test mode"

        self.max_seq_len = max_seq_len
        self.mode        = mode

        # --- Mappings ---
        with open(source_mapping_path,  'r') as f: self.source_mapping = json.load(f)
        with open(target_mapping_path,  'r') as f: self.target_mapping = json.load(f)
        with open(user_mapping_path,    'r') as f: self.user_mapping   = json.load(f)

        # --- Source history with timestamps (leakage prevention) ---
        source_df = pd.read_csv(source_inter_path, sep='\t')
        self.user_source_history = self._build_history_with_ts(
            source_df, self.source_mapping
        )

        # --- Build samples ---
        self.samples = []

        if mode == 'train':
            self._build_train_samples(target_inter_path)
        else:
            self._build_eval_samples(target_inter_path, train_target_inter_path)

    # ------------------------------------------------------------------
    # History builder
    # ------------------------------------------------------------------

    def _build_history_with_ts(self, df, item_mapping):
        """
        Returns {user_id: [(item_idx, timestamp), ...]} sorted by timestamp.
        item_idx is 1-indexed (0 = padding).
        """
        history = {}
        df_sorted = df.sort_values(by=['user_id:token', 'timestamp:float'])
        for user_str, group in df_sorted.groupby('user_id:token'):
            user_str = str(user_str)
            if user_str not in self.user_mapping:
                continue
            items = [
                (item_mapping[str(tid)] + 1, float(ts))
                for tid, ts in zip(
                    group['item_id:token'].values,
                    group['timestamp:float'].values
                )
                if str(tid) in item_mapping
            ]
            history[self.user_mapping[user_str]] = items
        return history

    # ------------------------------------------------------------------
    # Sample builders
    # ------------------------------------------------------------------

    def _build_train_samples(self, target_inter_path):
        target_df = pd.read_csv(target_inter_path, sep='\t')
        grouped   = (
            target_df.sort_values('timestamp:float')
                     .groupby('user_id:token')
        )

        for user_str, group in grouped:
            user_str = str(user_str)
            if user_str not in self.user_mapping:
                continue
            user_id = self.user_mapping[user_str]

            rows = [
                (self.target_mapping[str(tid)] + 1, float(ts))
                for tid, ts in zip(
                    group['item_id:token'].values,
                    group['timestamp:float'].values
                )
                if str(tid) in self.target_mapping
            ]

            for i in range(1, len(rows)):
                target_id, cutoff_ts = rows[i]
                target_history = [mid for mid, _ in rows[:i]]
                self.samples.append({
                    'user_id'        : user_id,
                    'target_history' : target_history,
                    'target_item_id' : target_id,
                    'cutoff_ts'      : cutoff_ts,
                })

        print(f"[Train] {len(self.samples)} sliding-window samples generated.")

    def _build_eval_samples(self, target_inter_path, train_inter_path):
        train_df  = pd.read_csv(train_inter_path, sep='\t')
        target_df = pd.read_csv(target_inter_path, sep='\t')

        # Train target history (for context)
        train_target_history = {}
        for user_str, group in (
            train_df.sort_values('timestamp:float').groupby('user_id:token')
        ):
            user_str = str(user_str)
            if user_str not in self.user_mapping:
                continue
            uid   = self.user_mapping[user_str]
            items = [
                (self.target_mapping[str(tid)] + 1, float(ts))
                for tid, ts in zip(
                    group['item_id:token'].values,
                    group['timestamp:float'].values
                )
                if str(tid) in self.target_mapping
            ]
            train_target_history[uid] = items

        for _, row in target_df.iterrows():
            u_str = str(row['user_id:token'])
            i_str = str(row['item_id:token'])
            if u_str not in self.user_mapping or i_str not in self.target_mapping:
                continue

            user_id   = self.user_mapping[u_str]
            target_id = self.target_mapping[i_str] + 1
            cutoff_ts = float(row['timestamp:float'])
            history   = train_target_history.get(user_id, [])

            self.samples.append({
                'user_id'        : user_id,
                'target_history' : [mid for mid, _ in history],
                'target_item_id' : target_id,
                'cutoff_ts'      : cutoff_ts,
            })

        print(f"[{self.mode}] {len(self.samples)} eval samples generated.")

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample  = self.samples[idx]
        user_id = sample['user_id']
        cutoff  = sample['cutoff_ts']

        # Source seq: only interactions before cutoff
        all_source = self.user_source_history.get(user_id, [])
        source_seq = [
            sid for sid, ts in all_source if ts < cutoff
        ][-self.max_seq_len:]

        target_seq = sample['target_history'][-self.max_seq_len:]

        # Padding
        source_seq = source_seq + [0] * (self.max_seq_len - len(source_seq))
        target_seq = target_seq + [0] * (self.max_seq_len - len(target_seq))

        # Masks (True = padding)
        source_mask = [x == 0 for x in source_seq]
        target_mask = [x == 0 for x in target_seq]

        return {
            'user_id'        : torch.tensor(user_id,                 dtype=torch.long),
            'target_item_id' : torch.tensor(sample['target_item_id'],  dtype=torch.long),
            'source_seq'     : torch.tensor(source_seq,                dtype=torch.long),
            'target_seq'     : torch.tensor(target_seq,                dtype=torch.long),
            'source_mask'    : torch.tensor(source_mask,               dtype=torch.bool),
            'target_mask'    : torch.tensor(target_mask,               dtype=torch.bool),
        }