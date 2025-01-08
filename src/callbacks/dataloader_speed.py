import time
from composer.core import Callback, State
from composer.loggers import Logger
import glob

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

# import json
# from pathlib import Path
# import numpy as np
# import torch
# import zstandard as zstd_standard
# from composer.core import Callback, State
# from typing import Dict, Any, Optional
# from composer.loggers import Logger

# class StepFolderCallback(Callback):
#     """Creates checkpoint folders for saving data."""
    
#     def __init__(
#         self,
#         data_save_folder: str,
#         model_save_folder: str,
#         checkpoint_format: str = "checkpoint_{checkpoint:06d}",
#     ):
#         self.base_save_dir = Path(data_save_folder)
#         self.checkpoint_format = checkpoint_format
#         self.base_save_dir.mkdir(parents=True, exist_ok=True)
#         self.current_checkpoint = 0
#         self.model_save_folder = model_save_folder
#         self.number_of_checkpoints = 0
#         print(f"Initializing StepFolderCallback with model_save_folder: {self.model_save_folder}")
#         checkpoint_folder = self.base_save_dir / self.checkpoint_format.format(checkpoint=self.current_checkpoint)
#         checkpoint_folder.mkdir(exist_ok=True)
    
#     def _update_save_folder(self, state: State) -> None:
#         """Updates the model's save folder for current checkpoint."""        
#         checkpoint_folder = self.base_save_dir / self.checkpoint_format.format(checkpoint=self.current_checkpoint)
#         checkpoint_folder.mkdir(exist_ok=True)
            
#         # Update the model's save folder to point to the checkpoint folder
#         if hasattr(state.model, 'module'):  # Check if it's wrapped in DDP
#             state.model.module.config.data_save_folder = str(checkpoint_folder)
#         else:
#             state.model.config.data_save_folder = str(checkpoint_folder)
#         print(f"Updated save folder to: {checkpoint_folder}")

#     def fit_start(self, state: State, logger: Optional[Logger] = None) -> None:
#         """Ensure folder is set up before any training begins."""
#         if self.current_checkpoint == 0:
#             self._update_save_folder(state)

#     def get_number_of_checkpoints(self) -> int:
#         """Get the number of checkpoints."""
#         print(f"Checking for files like {self.model_save_folder}/*.pt")
#         number_of_checkpoints = len(glob.glob(f"{self.model_save_folder}/*.pt"))
#         print(f"Number of checkpoints: {number_of_checkpoints}")
#         return number_of_checkpoints

#     def batch_checkpoint(self, state: State, batch_idx: int) -> None:
#         """Called after each batch checkpoint to update the save folder."""
#         if self.number_of_checkpoints != self.get_number_of_checkpoints():
#             self.current_checkpoint += 1
#             self.number_of_checkpoints = self.get_number_of_checkpoints()
#             print(f"In batch_checkpoint, current_checkpoint: {self.current_checkpoint}")
#             self._update_save_folder(state)

#     def iteration_checkpoint(self, state: State, batch_idx: int) -> None:
#         """Called after each iteration checkpoint to update the save folder."""
#         self.current_checkpoint += 1
#         print(f"In iteration_checkpoint, current_checkpoint: {self.current_checkpoint}")
#         self._update_save_folder(state)

#     def epoch_checkpoint(self, state: State, batch_idx: int) -> None:
#         """Called after each epoch checkpoint to update the save folder."""
#         self.current_checkpoint += 1
#         print(f"In epoch_checkpoint, current_checkpoint: {self.current_checkpoint}")
#         self._update_save_folder(state)


# class BatchSaverCallback(Callback):
#     """Saves batches at every step using model's save_folder."""
    
#     def __init__(
#         self,
#         compression_level: int = 3,
#         steps_per_file: int = 10
#     ):
#         self.compression_level = compression_level
#         self.steps_per_file = steps_per_file
#         self.current_file = None
#         self.current_file_group = -1
#         self.cctx = zstd_standard.ZstdCompressor(level=compression_level)
        
#     def _convert_tensor_to_json(self, tensor: torch.Tensor) -> list:
#         """Convert a tensor to JSON-serializable format."""
#         return tensor.cpu().numpy().tolist()
    
#     def _save_batch(self, batch: torch.Tensor, state: State, step):
#         """Save a batch for the current step using model's save_folder."""
#         if not hasattr(state.model.config, 'data_save_folder'):
#             raise AttributeError("Model.config must have a data_save_folder attribute")
        
#         # Use the save folder directly - it's already pointing to the checkpoint folder
#         save_folder = Path(state.model.config.data_save_folder)
        
#         # Calculate which file group this step belongs to
#         file_group = step.value // self.steps_per_file
        
#         # Create new file if moving to new file group or if file is not open
#         if file_group != self.current_file_group or self.current_file is None:
#             if self.current_file is not None:
#                 self.current_file.close()
#                 self.current_file = None
                
#             start_step = file_group * self.steps_per_file
#             end_step = (file_group + 1) * self.steps_per_file - 1
            
#             filename = save_folder / f"batches_{start_step:06d}_{end_step:06d}.jsonl.zst"
#             print(f"Creating new batch file: {filename}")  # Debug print
            
#             try:
#                 raw_file = open(filename, 'wb')
#                 self.current_file = self.cctx.stream_writer(raw_file)
#                 self.current_file_group = file_group
#             except Exception as e:
#                 print(f"Error creating file {filename}: {e}")
#                 raise
        
#         try:
#             batch_size = batch.size(0)
            
#             # Save each sample in the batch
#             for i in range(batch_size):
#                 json_sample = {
#                     'input_ids': self._convert_tensor_to_json(batch[i]),
#                     'step': step.value,
#                 }
#                 json_str = json.dumps(json_sample) + '\n'
#                 self.current_file.write(json_str.encode('utf-8'))
#         except Exception as e:
#             print(f"Error writing to file: {e}")
#             raise

#     def batch_end(self, state: State, batch: Dict[str, torch.Tensor]) -> None:
#         """Called after each batch."""
#         if "input_ids" not in state.batch:
#             print("Warning: No input_ids found in batch")
#             return
            
#         input_ids = state.batch["input_ids"]
#         self._save_batch(input_ids, state, state.timestamp.batch)
    
#     def close(self, state: Optional[State] = None, logger: Optional[Logger] = None) -> None:
#         """Close any open files."""
#         if self.current_file is not None:
#             try:
#                 self.current_file.close()
#             except Exception as e:
#                 print(f"Error closing file: {e}")
#             self.current_file = None


### v2
# import json
# from pathlib import Path
# import numpy as np
# import torch
# import zstandard as zstd_standard
# from composer.core import Callback, State, Event, Time
# from typing import Dict, Any, Optional, Union, Callable, List
# from composer.loggers import Logger
# import threading
# from concurrent.futures import ThreadPoolExecutor
# import queue
# import os
# from composer.utils import create_interval_scheduler

# class StepFolderCallback(Callback):
#     """Creates checkpoint folders for saving data, synchronized with CheckpointSaver."""
    
#     def __init__(
#         self,
#         data_save_folder: str,
#         model_save_folder: str,
#         checkpoint_format: str = "checkpoint_{checkpoint:06d}",
#         save_interval: Union[Time, str, int, Callable[[State, Event], bool]] = "1ep",
#     ):
#         self.data_save_folder = Path(data_save_folder)
#         self.model_save_folder = Path(model_save_folder)
#         self.checkpoint_format = checkpoint_format
#         self.current_checkpoint = 0
        
#         # Create necessary directories
#         os.makedirs(self.data_save_folder, exist_ok=True)
#         os.makedirs(self.model_save_folder, exist_ok=True)
        
#         # Use same interval scheduler as CheckpointSaver
#         if not callable(save_interval):
#             self.save_interval = create_interval_scheduler(save_interval)
#         else:
#             self.save_interval = save_interval
#         self.last_checkpoint_batch = None
        
#         # Set initial folder
#         self._create_checkpoint_folder(self.current_checkpoint)
    
#     def _create_checkpoint_folder(self, checkpoint_num: int) -> Path:
#         """Create checkpoint folder."""
#         folder = self.data_save_folder / self.checkpoint_format.format(checkpoint=checkpoint_num)
#         os.makedirs(folder, exist_ok=True)
#         return folder
    
#     def _update_save_folder(self, state: State) -> None:
#         """Updates the model's save folder for current checkpoint."""
#         checkpoint_folder = self._create_checkpoint_folder(self.current_checkpoint)
#         model = getattr(state.model, 'module', state.model)
#         model.config.data_save_folder = str(checkpoint_folder)

#     def fit_start(self, state: State, logger: Optional[Logger] = None) -> None:
#         """Set initial save folder."""
#         self._update_save_folder(state)
            
#     def batch_checkpoint(self, state: State, logger: Logger) -> None:
#         """Check for checkpoint using same logic as CheckpointSaver."""
#         if (self.save_interval(state, Event.BATCH_CHECKPOINT) and 
#             self.last_checkpoint_batch != state.timestamp.batch):
#             self.current_checkpoint += 1
#             self.last_checkpoint_batch = state.timestamp.batch
#             self._update_save_folder(state)
            
#     def iteration_checkpoint(self, state: State, logger: Logger) -> None:
#         """Check for checkpoint using same logic as CheckpointSaver."""
#         if (self.save_interval(state, Event.ITERATION_CHECKPOINT) and
#             self.last_checkpoint_batch != state.timestamp.batch):
#             self.current_checkpoint += 1
#             self.last_checkpoint_batch = state.timestamp.batch
#             self._update_save_folder(state)
        
#     def epoch_checkpoint(self, state: State, logger: Logger) -> None:
#         """Check for checkpoint using same logic as CheckpointSaver."""
#         if (self.save_interval(state, Event.EPOCH_CHECKPOINT) and
#             self.last_checkpoint_batch != state.timestamp.batch):
#             self.current_checkpoint += 1
#             self.last_checkpoint_batch = state.timestamp.batch
#             self._update_save_folder(state)

# class BatchSaverCallback(Callback):
#     """Efficiently saves batches to compressed JSONL files without async complexity."""
    
#     def __init__(
#         self,
#         steps_per_file: int = 100,
#         compression_level: int = 3
#     ):
#         self.steps_per_file = steps_per_file
#         self.compression_level = compression_level
#         self.current_file_data = []
#         self.current_file_group = -1
        
#         # Initialize compressor once
#         self.compressor = zstd_standard.ZstdCompressor(level=compression_level)
    
#     def _write_current_file(self, save_folder: Path) -> None:
#         """Write accumulated batches to a compressed file."""
#         if not self.current_file_data:
#             return
            
#         start_step = self.current_file_group * self.steps_per_file
#         end_step = start_step + self.steps_per_file - 1
#         filename = save_folder / f"batches_{start_step:06d}_{end_step:06d}.jsonl.zst"
        
#         # Write directly to disk with compression
#         with open(filename, 'wb') as f:
#             with self.compressor.stream_writer(f) as compressor:
#                 for item in self.current_file_data:
#                     json_str = json.dumps(item) + '\n'
#                     compressor.write(json_str.encode('utf-8'))
        
#         # Clear the current file data
#         self.current_file_data = []
    
#     def batch_end(self, state: State, batch: Dict[str, torch.Tensor]) -> None:
#         """Process each batch and write files when enough batches accumulate."""
#         if "input_ids" not in state.batch:
#             return
            
#         save_folder = Path(state.model.config.data_save_folder)
#         input_ids = state.batch["input_ids"]
#         step = state.timestamp.batch.value
        
#         # Check if we need to start a new file
#         file_group = step // self.steps_per_file
#         if file_group != self.current_file_group:
#             if self.current_file_group >= 0:
#                 self._write_current_file(save_folder)
#             self.current_file_group = file_group
        
#         # Convert batch to list of samples and add to current file data
#         batch_data = [
#             {
#                 'input_ids': tensor.cpu().numpy().tolist(),
#                 'step': step
#             }
#             for tensor in input_ids
#         ]
#         self.current_file_data.extend(batch_data)
    
#     def close(self, state: Optional[State] = None, logger: Optional[Logger] = None) -> None:
#         """Write any remaining data when training ends."""
#         if state and hasattr(state.model, 'config'):
#             save_folder = Path(state.model.config.data_save_folder)
#             self._write_current_file(save_folder)