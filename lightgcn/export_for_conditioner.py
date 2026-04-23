import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import torch

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from data import RecommendationDataset, load_interactions_from_file
from models import LightGCN


KNOWN_AUTO_DOMAINS = ("movie", "book", "music")


def resolve_input_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def infer_domain_from_stem(stem: str) -> str | None:
    name = stem.lower()
    for domain in KNOWN_AUTO_DOMAINS:
        if domain in name:
            return domain
    return None


def collect_domain_item_files(args: argparse.Namespace) -> Dict[str, List[Path]]:
    domain_files: Dict[str, List[Path]] = defaultdict(list)

    def add_domain_file(domain: str, file_path: Path) -> None:
        if file_path not in domain_files[domain]:
            domain_files[domain].append(file_path)

    if args.domain_items:
        for spec in args.domain_items:
            if "=" not in spec:
                raise ValueError(f"Invalid --domain_items value '{spec}'. Expected domain=path")
            domain_name, file_part = spec.split("=", 1)
            domain = domain_name.strip().lower()
            if not domain:
                raise ValueError(f"Invalid --domain_items value '{spec}': empty domain")
            file_path = resolve_input_path(file_part.strip())
            if not file_path.exists():
                raise FileNotFoundError(f"Domain source file not found: {file_path}")
            add_domain_file(domain, file_path)
        return dict(domain_files)

    dataset_dir = resolve_input_path(args.dataset_path).parent
    if dataset_dir.exists():
        for file_path in sorted(dataset_dir.glob("*.inter")):
            stem = file_path.stem.lower()
            if "combined" in stem:
                continue
            domain = infer_domain_from_stem(stem)
            if domain is not None:
                add_domain_file(domain, file_path)

    if args.test_path:
        test_path = resolve_input_path(args.test_path)
        if test_path.exists():
            domain = infer_domain_from_stem(test_path.stem.lower())
            if domain is not None:
                add_domain_file(domain, test_path)

    return dict(domain_files)


def build_domain_item_indices(item_mapping: Dict[Any, int], domain_item_files: Dict[str, List[Path]]) -> Dict[str, torch.Tensor]:
    domain_indices: Dict[str, torch.Tensor] = {}
    if not item_mapping or not domain_item_files:
        return domain_indices

    for domain, files in sorted(domain_item_files.items()):
        item_tokens = set()
        for file_path in files:
            rows = load_interactions_from_file(str(file_path))
            for row in rows:
                if len(row) >= 2:
                    item_tokens.add(row[1])

        mapped = sorted({item_mapping[token] for token in item_tokens if token in item_mapping})
        if mapped:
            domain_indices[domain] = torch.tensor(mapped, dtype=torch.long)

    return domain_indices


def load_dataset(args: argparse.Namespace) -> RecommendationDataset:
    if args.test_path:
        return RecommendationDataset.from_separate_files(
            train_filepath=args.dataset_path,
            test_filepath=args.test_path,
            max_rows=args.max_rows,
        )
    return RecommendationDataset.from_file(
        filepath=args.dataset_path,
        test_ratio=args.test_ratio,
        split_strategy=args.split_strategy,
        seed=args.seed,
        max_rows=args.max_rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export LightGCN embeddings in collaborator format")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--test_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default="outputs/lightgcn_embeddings.pt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", choices=["cpu", "cuda"])
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--domain_items", action="append", default=[])
    parser.add_argument("--split_strategy", type=str, default="leave_one_out", choices=["random", "leave_one_out", "all_train"])
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    config = checkpoint.get("config", {})

    dataset = load_dataset(args)
    n_users_ckpt = state_dict["user_embedding.weight"].shape[0]
    n_items_ckpt = state_dict["item_embedding.weight"].shape[0]
    if dataset.n_users != n_users_ckpt or dataset.n_items != n_items_ckpt:
        raise ValueError(
            "Dataset cardinality does not match checkpoint.\n"
            f"  checkpoint: users={n_users_ckpt}, items={n_items_ckpt}\n"
            f"  dataset:    users={dataset.n_users}, items={dataset.n_items}"
        )

    domain_item_files = collect_domain_item_files(args)
    domain_item_indices = build_domain_item_indices(dataset.item_mapping or {}, domain_item_files)
    export_domains = sorted(domain_item_indices.keys())
    if not export_domains:
        raise ValueError("No domain item sets were resolved for export.")

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = LightGCN(
        n_users=dataset.n_users,
        n_items=dataset.n_items,
        embedding_dim=int(config.get("embedding_dim")),
        n_layers=int(config.get("n_layers")),
        dropout=float(config.get("dropout", 0.0)),
        reg_weight=float(config.get("reg_weight", 1e-4)),
    ).to(device)
    model.load_state_dict(state_dict)
    model.set_adjacency_matrix(dataset.interaction_pairs, device)
    model.eval()

    with torch.no_grad():
        user_emb, item_emb = model.get_embeddings()
        user_emb = user_emb.cpu()
        item_emb = item_emb.cpu()

    payload = {
        "model": str(config.get("model", "lightgcn")),
        "ckpt_path": str(ckpt_path.resolve()),
        "hidden": int(user_emb.shape[1]),
        "embeddings": {"user": user_emb},
        "num_nodes": {"user": int(user_emb.shape[0])},
        "metadata": {
            "run_id": checkpoint.get("run_id", ""),
            "epoch": int(checkpoint.get("epoch", -1)),
            "dataset_path": args.dataset_path,
            "test_path": args.test_path or "",
            "domains": export_domains,
        },
    }

    for domain in export_domains:
        item_indices = domain_item_indices[domain]
        payload["embeddings"][domain] = item_emb[item_indices]
        payload["num_nodes"][domain] = int(item_indices.numel())

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)

    print(f"[SAVED] {output_path}")
    print(f"  user embeddings: {tuple(user_emb.shape)}")
    for domain in export_domains:
        count = int(domain_item_indices[domain].numel())
        print(f"  {domain} embeddings: ({count}, {user_emb.shape[1]})")


if __name__ == "__main__":
    main()
