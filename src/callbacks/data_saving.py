import os
from pathlib import Path
from typing import Dict, Optional, Union, Callable, List
import numpy as np
import torch
import pyarrow as pa
import pyarrow.parquet as pq
from composer.core import Callback, State, Event, Time
from composer.loggers import Logger
from composer.utils import create_interval_scheduler
import logging
from concurrent.futures import ThreadPoolExecutor, Future
from threading import Lock
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedDataSaver(Callback):
    """
    Balanced callback for efficient data saving with reasonable file sizes.
    Optimized for both CPU and I/O performance.
    """
    
    def __init__(
        self,
        data_save_folder: str,
        model_save_folder: str,
        batches_per_file: int = 300,
        checkpoint_format: str = "checkpoint_{checkpoint:06d}",
        save_interval: Union[Time, str, int, Callable[[State, Event], bool]] = "1ep",
        num_workers: int = 10,
        compression_level: int = 10,
    ):
        logger.info("Initializing OptimizedDataSaver")
        self.data_save_folder = Path(data_save_folder)
        self.model_save_folder = Path(model_save_folder)
        self.checkpoint_format = checkpoint_format
        self.current_checkpoint = 1 # start at 1 since steps are 1 indexed as are models.
        self.batches_per_file = batches_per_file
        self.compression_level = compression_level
        self.max_pending_writes = 10
        
        # Create directories
        os.makedirs(self.data_save_folder, exist_ok=True)
        os.makedirs(self.model_save_folder, exist_ok=True)
        
        # Checkpoint tracking
        if not callable(save_interval):
            self.save_interval = create_interval_scheduler(save_interval)
        else:
            self.save_interval = save_interval
        self.last_checkpoint_batch = None
        
        # Batch accumulation
        self.accumulated_input_ids = []
        self.accumulated_steps = []
        self.current_file_group = 0
        self.batches_in_current_file = 0
        
        # Threading setup
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.write_lock = Lock()
        self.pending_writes: List[Future] = []
        
        # Metrics
        self.total_batches_processed = 0
        self.total_files_written = 0
        self.start_time = time.time()
        self.write_times = []  # Track write durations

        # Add new thread management parameters
        self.min_write_interval = 30 # Minimum seconds between writes
        self.last_write_time = 0.0
        self.write_queue_lock = Lock()
        self.max_pending_writes = 2  # Reduce from default
        
        logger.info(
            f"Configuration:\n"
            f"- Batches per file: {batches_per_file}\n"
            f"- Compression level: {compression_level}\n"
            f"- Number of workers: {num_workers}"
        )
    
    def _write_parquet_file(
        self, 
        input_ids_array: np.ndarray, 
        steps_array: np.ndarray,
        save_folder: Path, 
        file_group: int
    ) -> None:
        """Write accumulated arrays with optimized PyArrow conversion."""
        if len(steps_array) == 0:
            return
                
        if "checkpoint" not in str(save_folder):
            checkpoint_folder = save_folder / f"checkpoint_{self.current_checkpoint:06d}"
        else:
            checkpoint_folder = save_folder
            
        os.makedirs(checkpoint_folder, exist_ok=True)
        
        start_step = min(steps_array)
        end_step = max(steps_array)
        filename = checkpoint_folder / f"steps_{start_step:06d}_to_{end_step:06d}.parquet"            
        try:
            steps = pa.array(steps_array.astype(np.uint64))

            input_ids_list = [row.tolist() for row in input_ids_array]
            input_ids = pa.array(input_ids_list, type=pa.list_(pa.uint16()))
            
            # Create table directly from arrays
            table = pa.Table.from_arrays(
                [input_ids, steps],
                names=['input_ids', 'step'],
                metadata={
                    b'sequence_length': str(input_ids_array.shape[1]).encode()
                }
            )
            
            start_time = time.time()    
            pq.write_table(
                table,
                filename,
                compression='ZSTD',
                compression_level=self.compression_level,
                use_dictionary=False,  # Disable dictionary encoding for input_ids
                write_statistics=False,  # Disable statistics for faster writes
                data_page_size=8*1024*1024,  # 8MB pages for better I/O
                row_group_size=100000,  # Larger row groups
                use_byte_stream_split=False,  # Disable byte stream splitting
                filesystem=None  # Use native filesystem
            )
            
            duration = time.time() - start_time
            file_size = os.path.getsize(filename)
            write_speed = file_size / (1024 * 1024 * duration)  # MB/s
            
            with self.write_lock:
                self.total_files_written += 1
                self.write_times.append(duration)
            
            logger.info(
                f"Wrote {filename.name}: "
                f"{len(steps_array)} samples, "
                f"{file_size / (1024*1024):.1f}MB, "
                f"{write_speed:.1f}MB/s, "
                f"{duration:.2f}s"
            )
            
            logger.info(f"Successfully created file at {filename}")
        except Exception as e:
            logger.error(f"Error writing file {filename}: {str(e)}", exc_info=True)
            raise
    
    def batch_end(self, state: State, batch: Dict[str, torch.Tensor]) -> None:
        """Process batch data with controlled write intervals."""
        current_time = time.time()
        logger.debug(f"Processing batch at time {current_time}, checkpoint {self.current_checkpoint}")
    
        # Add rate limiting at the start
        current_time = time.time()
        with self.write_queue_lock:
            time_since_last_write = current_time - self.last_write_time
            if time_since_last_write < self.min_write_interval:
                time.sleep(max(0, self.min_write_interval - time_since_last_write))
        
        # Wait for pending writes to complete if we're at the limit
        while len(self.pending_writes) >= self.max_pending_writes:
            # Clean up completed writes
            self.pending_writes = [f for f in self.pending_writes if not f.done()]
            if len(self.pending_writes) >= self.max_pending_writes:
                logger.info(f"Waiting for {len(self.pending_writes)} pending writes to complete...")
                time.sleep(0.5)  # Increased sleep time to reduce CPU usage
        
        if "input_ids" not in state.batch:
            return

        save_folder = Path(state.model.config.data_save_folder)
        input_ids = state.batch["input_ids"]
        step = state.timestamp.batch.value
        
        # Convert tensors efficiently
        input_ids_np = input_ids.cpu().numpy()
        if len(input_ids_np.shape) != 2:
            raise ValueError(f"Expected input_ids to be 2D tensor, got shape {input_ids_np.shape}")
        
        # Add to accumulation (preserving 2D structure)
        self.accumulated_input_ids.append(input_ids_np)
        self.accumulated_steps.extend([step] * len(input_ids))
        self.batches_in_current_file += 1
        
        # Log first batch shape for debugging
        if self.batches_in_current_file == 1:
            logger.info(f"First batch shape: {input_ids_np.shape}")
            logger.info(f"Starting file group {self.current_file_group} in checkpoint {self.current_checkpoint}")
        
        # Write when we hit our batch limit
        if self.batches_in_current_file >= self.batches_per_file:
            logger.info(f"Reached {self.batches_per_file} batches, writing file...")
            self._flush_accumulated_data(save_folder)
        
        # Update metrics
        with self.write_lock:
            self.total_batches_processed += 1
            if self.total_batches_processed % self.batches_per_file == 0:
                elapsed = time.time() - self.start_time
                rate = self.total_batches_processed / elapsed
                avg_write_time = np.mean(self.write_times) if self.write_times else 0
                logger.info(
                    f"Progress: {self.total_batches_processed} batches "
                    f"({rate:.1f} batches/s), "
                    f"avg write time: {avg_write_time:.2f}s, "
                    f"files written: {self.total_files_written}"
                )
            
        # Write when we hit our batch limit    
        if self.batches_in_current_file >= self.batches_per_file:
            logger.info(
                f"Initiating write for checkpoint {self.current_checkpoint}, "
                f"file group {self.current_file_group}"
            )
            with self.write_queue_lock:
                self._flush_accumulated_data(Path(state.model.config.data_save_folder))
                self.last_write_time = time.time()

    def _flush_accumulated_data(self, save_folder: Path) -> None:
        """Flush accumulated data with controlled write rate."""
        if not self.accumulated_steps:
            return
            
        # Convert accumulated data
        input_ids_array = np.concatenate(self.accumulated_input_ids)
        steps_array = np.array(self.accumulated_steps)
        
        # Start async write with rate limiting
        future = self.executor.submit(
            self._write_parquet_file,
            input_ids_array,
            steps_array,
            save_folder,
            self.current_file_group
        )
        
        self.pending_writes.append(future)
        
        # Clear accumulation buffers
        self.accumulated_input_ids = []
        self.accumulated_steps = []
        self.batches_in_current_file = 0
        self.current_file_group += 1
        
        # Clean up completed writes
        self.pending_writes = [f for f in self.pending_writes if not f.done()]
    
    def _wait_for_pending_writes(self) -> None:
        """Wait for all pending writes to complete."""
        if not self.pending_writes:
            return
            
        logger.info(f"Waiting for {len(self.pending_writes)} pending writes...")
        start_time = time.time()
        
        for future in self.pending_writes:
            try:
                future.result()
            except Exception as e:
                logger.error(f"Write failed: {str(e)}", exc_info=True)
        
        self.pending_writes.clear()
        duration = time.time() - start_time
        logger.info(f"All writes completed in {duration:.2f}s")
    
    def _handle_checkpoint(self, state: State) -> None:
        """Handle checkpoint transitions."""
        # print the current batch and token count
        logger.info(f"Checkpoint transition: Current batch: {state.timestamp.batch}, token count: {state.timestamp.token}")
        if self.last_checkpoint_batch != state.timestamp.batch:
            logger.info(f"Creating checkpoint {self.current_checkpoint + 1}")
            save_folder = Path(state.model.config.data_save_folder)
            self._flush_accumulated_data(save_folder)
            self._wait_for_pending_writes()  # Ensure all writes complete before checkpoint
            self.current_checkpoint += 1
            self.last_checkpoint_batch = state.timestamp.batch
            checkpoint_folder = self._create_checkpoint_folder(self.current_checkpoint)
            model = getattr(state.model, 'module', state.model)
            model.config.data_save_folder = str(checkpoint_folder)
    
    def _create_checkpoint_folder(self, checkpoint_num: int) -> Path:
        """Create checkpoint folder."""
        folder = self.data_save_folder / self.checkpoint_format.format(checkpoint=checkpoint_num)
        os.makedirs(folder, exist_ok=True)
        return folder
    
    def batch_checkpoint(self, state: State, logger: Logger) -> None:
        if self.save_interval(state, Event.BATCH_CHECKPOINT):
            self._handle_checkpoint(state)
            
    def iteration_checkpoint(self, state: State, logger: Logger) -> None:
        if self.save_interval(state, Event.ITERATION_CHECKPOINT):
            self._handle_checkpoint(state)
        
    def epoch_checkpoint(self, state: State, logger: Logger) -> None:
        if self.save_interval(state, Event.EPOCH_CHECKPOINT):
            self._handle_checkpoint(state)
    
    def close(self, state: Optional[State] = None, logger: Optional[Logger] = None) -> None:
        """Clean up and log statistics."""
        logger.info("Shutting down OptimizedDataSaver...")
        
        try:
            # Write remaining data
            if state and hasattr(state.model, 'config'):
                save_folder = Path(state.model.config.data_save_folder)
                self._flush_accumulated_data(save_folder)
            
            # Wait for pending writes
            self._wait_for_pending_writes()
            
            # Shutdown executor
            self.executor.shutdown(wait=True)
            
            # Log final statistics
            elapsed = time.time() - self.start_time
            avg_write_time = np.mean(self.write_times) if self.write_times else 0
            logger.info(
                f"OptimizedDataSaver finished:\n"
                f"- Total batches processed: {self.total_batches_processed}\n"
                f"- Total files written: {self.total_files_written}\n"
                f"- Average write time: {avg_write_time:.2f}s\n"
                f"- Average processing rate: {self.total_batches_processed / elapsed:.1f} batches/s\n"
                f"- Total runtime: {elapsed:.1f}s"
            )
            
        except Exception as e:
            logger.error("Error during shutdown:", exc_info=True)
            raise