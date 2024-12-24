import time
from composer.core import Callback, State
from composer.loggers import Logger

__all__ = ["DataloaderSpeedMonitor"]


class DataloaderSpeedMonitor(Callback):
    """Measure how long it takes to return a batch from the dataloader."""

    def before_dataloader(self, state: State, logger: Logger) -> None:
        del logger  # unused
        self.batch_start_time = time.time_ns()

    def after_dataloader(self, state: State, logger: Logger) -> None:
        self.batch_serve_time = time.time_ns() - self.batch_start_time
        logger.log_metrics(
            {
                "throughput/batch_serve_time_ns": self.batch_serve_time,
                "throughput/batch_serve_time_ms": self.batch_serve_time / 1e6,
            }
        )

import json
from pathlib import Path
import numpy as np
import torch
import zstandard as zstd_standard
from composer.core import Callback, State
from typing import Dict, Any, Optional


class BatchSaverCallback(Callback):
    """Saves batches at every step using model's save_folder, creating new files every 10 steps."""
    
    def __init__(
        self,
        compression_level: int = 3,
        steps_per_file: int = 10
    ):
        self.compression_level = compression_level
        self.steps_per_file = steps_per_file
        self.current_file = None
        self.current_file_group = -1
        self.cctx = zstd_standard.ZstdCompressor(level=compression_level)
        
    def _convert_sample_to_json(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a sample with torch tensors or numpy arrays to JSON-serializable format."""
        converted = {}
        for key, value in sample.items():
            if isinstance(value, (torch.Tensor, np.ndarray)):
                converted[key] = value.cpu().numpy().tolist() if isinstance(value, torch.Tensor) else value.tolist()
            else:
                converted[key] = value
        return converted
    
    def _save_batch(self, batch: Dict[str, torch.Tensor], state: State, step: int):
        """Save a batch for the current step using model's save_folder."""
        # Get current save folder from model
        if not hasattr(state.model, 'data_save_folder'):
            raise AttributeError("Model must have a save_folder attribute")
        
        save_folder = Path(state.model.data_save_folder)
        save_folder.mkdir(exist_ok=True)
            
        # Calculate which file group this step belongs to
        file_group = step // self.steps_per_file
        
        # Create new file if moving to new file group
        if file_group != self.current_file_group:
            if self.current_file is not None:
                self.current_file.close()
                
            filename = save_folder / f"batches_{file_group * self.steps_per_file:06d}_{(file_group + 1) * self.steps_per_file - 1:06d}.jsonl.zst"
            raw_file = open(filename, 'wb')
            self.current_file = self.cctx.stream_writer(raw_file)
            self.current_file_group = file_group
        
        # Save each sample in the batch
        for i in range(len(next(iter(batch.values())))):
            sample = {k: v[i] for k, v in batch.items()}
            json_sample = self._convert_sample_to_json(sample)
            # Add step number to sample
            json_sample['step'] = step
            json_str = json.dumps(json_sample) + '\n'
            self.current_file.write(json_str.encode('utf-8'))
    
    def batch_end(self, state: State, batch: Dict[str, torch.Tensor]) -> None:
        """Called after each batch."""
        self._save_batch(batch, state, state.timestamp.batch)
    
    def close(self):
        """Close any open files."""
        if self.current_file is not None:
            self.current_file.close()
            self.current_file = None


class StepFolderCallback(Callback):
    """Updates model's data_save_folder after each checkpoint."""
    
    def __init__(
        self,
        data_save_folder: str,
        folder_format: str = "step_{step:06d}",
    ):
        self.base_save_dir = Path(data_save_folder)
        self.folder_format = folder_format
        self.base_save_dir.mkdir(parents=True, exist_ok=True)
        self.current_step = -1
    
    def _update_save_folder(self, state: State, step: int) -> None:
        """Updates the model's save folder."""
        if step != self.current_step:
            if not hasattr(state.model, 'data_save_folder'):
                raise AttributeError("Model must have a data_save_folder attribute")
                
            new_folder = self.base_save_dir / self.folder_format.format(step=step)
            new_folder.mkdir(exist_ok=True)
            
            # Update the model's save folder
            state.model.data_save_folder = str(new_folder)
            self.current_step = step
    
    def batch_checkpoint(self, state: State, batch_idx: int) -> None:
        """Called after each checkpoint to update the save folder."""
        self._update_save_folder(state, state.timestamp.batch)

    def fit_end(self, state: State) -> None:
        """Called at the end of training."""
        self._update_save_folder(state, state.timestamp.batch)