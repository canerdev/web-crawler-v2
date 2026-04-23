"""
Results logging and model checkpointing utilities.
"""

import re
import csv
import json
import hashlib
import torch
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional


def generate_run_id(model_name: str, dataset_name: str, config: Dict) -> str:
    """Generate unique run ID from timestamp + config hash."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # Hash key config params for uniqueness
    config_str = json.dumps({
        'emb': config.get('embedding_dim'),
        'layers': config.get('n_layers'),
        'lr': config.get('learning_rate'),
        'reg': config.get('reg_weight'),
        'batch': config.get('batch_size'),
    }, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:6]
    return f"{model_name}_{dataset_name}_{timestamp}_{config_hash}"


def parse_kcore_from_path(path: str) -> Optional[int]:
    """
    Extract k-core value from file path (checks both folder and filename).
    
    Supports patterns like:
    - data/data/5-core/amazon/Books.csv -> 5 (from folder)
    - data/10-core/Books.csv -> 10 (from folder)
    - Books_k5.csv -> 5 (from filename)
    - Books_k20.csv -> 20 (from filename)
    """
    # First, check folder path for patterns like "5-core", "10-core"
    folder_patterns = [
        r'(\d+)-core',     # 5-core, 10-core, 20-core
        r'(\d+)core',      # 5core, 10core
        r'kcore(\d+)',     # kcore5, kcore10
        r'k(\d+)/',        # k5/, k10/ as folder
    ]
    
    for pattern in folder_patterns:
        match = re.search(pattern, path, re.IGNORECASE)
        if match:
            return int(match.group(1))
    
    # Fall back to filename patterns
    filename = Path(path).stem if path else ""
    filename_patterns = [
        r'_k(\d+)$',           # _k5, _k10, _k20 at end
        r'_kcore(\d+)$',       # _kcore5, _kcore10
        r'_(\d+)core$',        # _5core, _10core
    ]
    
    for pattern in filename_patterns:
        match = re.search(pattern, filename, re.IGNORECASE)
        if match:
            return int(match.group(1))
    
    return None


class ResultsLogger:
    """Logs training results to CSV files."""
    
    COLUMNS = [
        'timestamp', 'run_id', 'model', 'dataset', 'k_core', 
        'n_users', 'n_items', 'n_train', 'n_test',
        'embedding_dim', 'n_layers', 'batch_size', 'learning_rate', 'reg_weight', 'n_neg',
        'n_epochs', 'best_epoch',
        'HR@10', 'HR@20',
        'NDCG@10', 'NDCG@20',
        'training_time_sec', 'checkpoint_path'
    ]
    
    def __init__(self, results_dir: str = 'results'):
        """Initialize logger with results directory."""
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def log_result(
        self,
        model_name: str,
        dataset_name: str,
        dataset_info: Dict[str, int],
        config: Dict[str, Any],
        metrics: Dict[str, float],
        best_epoch: int,
        training_time: float,
        run_id: str = "",
        checkpoint_path: str = "",
        dataset_path: str = ""
    ) -> str:
        """
        Log training result to CSV file.
        
        Args:
            model_name: Name of the model (e.g., 'LightGCN')
            dataset_name: Name of the dataset (basename, e.g., 'Books2023')
            dataset_info: Dict with n_users, n_items, n_train, n_test
            config: Training configuration dict
            metrics: Final evaluation metrics dict
            best_epoch: Best epoch number
            training_time: Total training time in seconds
            run_id: Unique run identifier (links to checkpoint)
            checkpoint_path: Path to saved checkpoint file
            notes: Optional notes
            dataset_path: Full path to dataset file (for k-core detection from folder)
            
        Returns:
            Path to the results file
        """
        # Auto-detect k_core from dataset path (folder) or config
        k_core = config.get('k_core') or parse_kcore_from_path(dataset_path or dataset_name)
        
        # Check for overlapping datasets (different from k-core filtering)
        is_overlapping = 'overlapping' in (dataset_path or '').lower()
        
        # Create results file path with appropriate subdirectory
        if is_overlapping:
            results_subdir = self.results_dir / 'overlapping'
            results_subdir.mkdir(parents=True, exist_ok=True)
            results_file = results_subdir / f"{dataset_name}.csv"
        elif k_core:
            results_subdir = self.results_dir / f"{k_core}-core"
            results_subdir.mkdir(parents=True, exist_ok=True)
            results_file = results_subdir / f"{dataset_name}.csv"
        else:
            results_file = self.results_dir / f"{dataset_name}.csv"
        file_exists = results_file.exists()
        
        # Prepare row data
        row = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'run_id': run_id,
            'model': model_name,
            'dataset': dataset_name,
            'k_core': k_core if k_core is not None else '',
            'n_users': dataset_info.get('n_users', ''),
            'n_items': dataset_info.get('n_items', ''),
            'n_train': dataset_info.get('n_train', ''),
            'n_test': dataset_info.get('n_test', ''),
            'embedding_dim': config.get('embedding_dim', ''),
            'n_layers': config.get('n_layers', ''),
            'batch_size': config.get('batch_size', ''),
            'learning_rate': config.get('learning_rate', ''),
            'reg_weight': config.get('reg_weight', ''),
            'n_neg': config.get('n_neg', 1),
            'n_epochs': config.get('n_epochs', ''),
            'best_epoch': best_epoch,
            'HR@10': metrics.get('HR@10', ''),
            'HR@20': metrics.get('HR@20', ''),
            'NDCG@10': metrics.get('NDCG@10', ''),
            'NDCG@20': metrics.get('NDCG@20', ''),
            'training_time_sec': f"{training_time:.1f}",
            'checkpoint_path': checkpoint_path
        }
        
        # Read existing rows (if any), then rewrite with current header + all rows
        existing_rows = []
        if file_exists:
            with open(results_file, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                existing_rows = list(reader)
        
        # Write header + all rows (always overwrites header to pick up column changes)
        with open(results_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.COLUMNS, extrasaction='ignore')
            writer.writeheader()
            for old_row in existing_rows:
                writer.writerow(old_row)
            writer.writerow(row)
        
        print(f"\n[RESULTS] Saved to: {results_file}")
        return str(results_file)


class CheckpointManager:
    """Manages model checkpoints during training with unique run IDs."""
    
    def __init__(
        self,
        checkpoint_dir: str = 'checkpoints',
        model_name: str = 'model',
        dataset_name: str = 'dataset',
        run_id: str = None,
        config: Dict = None,
        metadata: Dict = None
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            model_name: Name of the model for filename
            dataset_name: Name of the dataset for filename
            run_id: Unique run identifier (auto-generated if None)
            config: Config dict for run_id generation
            metadata: Additional metadata to store (dataset_paths, split_strategies, etc.)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.metadata = metadata or {}
        
        # Generate or use provided run_id
        if run_id:
            self.run_id = run_id
        elif config:
            self.run_id = generate_run_id(model_name, dataset_name, config)
        else:
            # Fallback: timestamp only
            self.run_id = f"{model_name}_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.best_metric = 0.0
        self.best_epoch = 0
        self._checkpoint_path = None
    
    def _get_checkpoint_path(self) -> Path:
        """Get checkpoint file path using run_id."""
        return self.checkpoint_dir / f"{self.run_id}.pt"
    
    def save_if_best(
        self,
        model: torch.nn.Module,
        metric_value: float,
        epoch: int,
        config: Dict,
        optimizer: torch.optim.Optimizer
    ) -> bool:
        """
        Save checkpoint if current metric is best so far.
        
        Args:
            model: PyTorch model
            metric_value: Current metric value (higher is better)
            epoch: Current epoch number
            config: Training config to save with checkpoint
            optimizer: Optimizer to save for resume capability
            
        Returns:
            True if saved as new best, False otherwise
        """
        # Require minimum improvement to avoid saving on floating-point noise
        min_improvement = 1e-5
        if metric_value > self.best_metric + min_improvement:
            self.best_metric = metric_value
            self.best_epoch = epoch
            
            checkpoint = {
                'run_id': self.run_id,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'metric_value': metric_value,
                'config': config,
                'optimizer_state_dict': optimizer.state_dict(),
                'metadata': self.metadata
            }
            
            # Save to file with unique run_id
            self._checkpoint_path = self._get_checkpoint_path()
            torch.save(checkpoint, self._checkpoint_path)
            
            return True
        return False
    
    def get_best_checkpoint_path(self) -> Optional[str]:
        """Get path to best checkpoint if exists."""
        path = self._get_checkpoint_path()
        return str(path) if path.exists() else None
