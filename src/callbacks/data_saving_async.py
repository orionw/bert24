import os
from pathlib import Path
from typing import Dict, Optional, Union, Callable, List, Set, Tuple
import numpy as np
import torch
import pyarrow as pa
import pyarrow.parquet as pq
from composer.core import Callback, State, Event, Time
from composer.loggers import Logger
from composer.utils import create_interval_scheduler
import logging
import asyncio
import aiofiles
from concurrent.futures import ThreadPoolExecutor
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedDataSaver(Callback):
    """
    Asynchronous callback for efficient data saving with sequential processing.
    Uses a queue to ensure strict ordering of batches and file writing.
    """
    
    def __init__(
        self,
        data_save_folder: str,
        model_save_folder: str,
        batches_per_file: int = 100,
        checkpoint_format: str = "checkpoint_{checkpoint:06d}",
        save_interval: Union[Time, str, int, Callable[[State, Event], bool]] = "1ep",
        num_workers: int = 10,
        compression_level: int = 1,
        min_write_interval: float = 12.0,  # 10s write + 2s buffer
        max_memory_usage: int = 1024 * 1024 * 1024 * 80,  # 80GB
    ):
        logger.info("Initializing AsyncOptimizedDataSaver")
        self.data_save_folder = Path(data_save_folder)
        self.model_save_folder = Path(model_save_folder)
        self.checkpoint_format = checkpoint_format
        self.current_checkpoint = 1
        self.batches_per_file = batches_per_file
        self.compression_level = compression_level
        self.max_memory_usage = max_memory_usage
        
        # Create directories
        os.makedirs(self.data_save_folder, exist_ok=True)
        os.makedirs(self.model_save_folder, exist_ok=True)
        
        # Checkpoint tracking
        if not callable(save_interval):
            self.save_interval = create_interval_scheduler(save_interval)
        else:
            self.save_interval = save_interval
        self.last_checkpoint_batch = None
        
        # Sequential processing setup
        self.batch_queue = asyncio.Queue()
        self.processing_task = None
        self.current_file_id = 1  # Tracks the current file number (1-based)
        
        # Batch accumulation
        self.accumulated_input_ids: List[np.ndarray] = []
        self.accumulated_steps: List[int] = []
        self.samples_in_current_file = 0
        
        # Async setup
        self.loop = asyncio.get_event_loop()
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        
        # Write pacing
        self.min_write_interval = min_write_interval
        self.last_write_time = time.time()
        
        # Metrics
        self.total_samples_processed = 0
        self.total_files_written = 0
        self.start_time = time.time()
        self.write_times: List[float] = []
        
        logger.info(
            f"Configuration:\n"
            f"- Batches per file: {batches_per_file}\n"
            f"- Compression level: {compression_level}\n"
            f"- Number of workers: {num_workers}\n"
            f"- Max memory usage: {max_memory_usage / (1024*1024):.1f}MB"
        )

    def batch_end(self, state: State, logger: Logger) -> None:
        """Handle batch end event from Composer framework."""
        try:
            # Process tensors before any async operations
            if isinstance(state.batch, dict) and "input_ids" in state.batch:
                # Detach and move to CPU immediately 
                input_ids = state.batch["input_ids"].detach().cpu()
                batch = {"input_ids": input_ids}
                
                # Add queue monitoring
                logger.info(f"Queue size before enqueue: {self.batch_queue.qsize()}")
                
                if not hasattr(self, 'loop') or self.loop.is_closed():
                    self.loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(self.loop)
                    self.processing_task = self.loop.create_task(self._process_queue())
                    logger.info("Started sequential batch processing task")
                
                self.loop.run_until_complete(self._enqueue_batch(state, batch))
                
                # Monitor task status
                if self.processing_task and self.processing_task.done():
                    if self.processing_task.exception():
                        logger.error(f"Processing task failed: {self.processing_task.exception()}")
                        raise self.processing_task.exception()
                
                # Log status every 100 batches
                if self.total_samples_processed % (100 * self.batches_per_file) == 0:
                    self.log_status()
                
                # Clear references
                del batch
                del input_ids
            
        except Exception as e:
            import traceback
            print(f"Error in batch_end: {str(e)} with traceback: {traceback.format_exc()}")
            logger.error(f"Error in batch_end: {str(e)}", exc_info=True)
            raise

    async def _process_batch(self, state: State, batch: Dict[str, torch.Tensor]):
        """Process a single batch."""
        try:
            input_ids = batch["input_ids"]
            # Verify we're not dealing with GPU tensors
            assert not input_ids.is_cuda, "GPU tensor detected in async processing queue!"
            batch_size = len(input_ids)
            
            # Convert tensors efficiently
            input_ids_np = input_ids.cpu().numpy()
            if len(input_ids_np.shape) != 2:
                raise ValueError(f"Expected input_ids to be 2D tensor, got shape {input_ids_np.shape}")
            
            # Calculate step range for this batch
            start_step = self.total_samples_processed + 1
            end_step = start_step + batch_size - 1
            
            # Add to accumulation
            self.accumulated_input_ids.append(input_ids_np)
            self.accumulated_steps.extend(range(start_step, end_step + 1))
            self.samples_in_current_file += batch_size
            self.total_samples_processed += batch_size
            
            # Log first batch in new file
            if len(self.accumulated_steps) == batch_size:
                logger.info(
                    f"Starting file {self.current_file_id} "
                    f"(batch shape: {input_ids_np.shape}, "
                    f"steps {start_step} to {end_step})"
                )
            
            # Write when we hit our batch limit
            if self.samples_in_current_file >= self.batches_per_file:
                logger.info(f"Triggering file write with {self.samples_in_current_file} samples")
                save_folder = Path(self.data_save_folder)  # Use the instance variable
                await self._write_file(save_folder)
                
                # Clear accumulation and increment file counter
                self.accumulated_input_ids = []
                self.accumulated_steps = []
                self.samples_in_current_file = 0
                self.current_file_id += 1
                
        except Exception as e:
            logger.error(f"Error in _process_batch: {str(e)}", exc_info=True)
            raise

    async def _write_file(self, save_folder: Path) -> None:
        """Write accumulated data to a Parquet file."""
        if not self.accumulated_steps:
            return
                
        start_step = min(self.accumulated_steps)
        end_step = max(self.accumulated_steps)
        
        try:
            checkpoint_folder = self.data_save_folder / f"checkpoint_{self.current_checkpoint:06d}"
            logger.info(f"Writing to checkpoint folder: {checkpoint_folder}")
            os.makedirs(checkpoint_folder, exist_ok=True)
            filename = checkpoint_folder / f"steps_{start_step:06d}_to_{end_step:06d}.parquet"
            
            current_time = time.time()
            time_since_last = current_time - self.last_write_time
            if time_since_last < self.min_write_interval:
                wait_time = self.min_write_interval - time_since_last
                logger.debug(f"Pacing: waiting {wait_time:.2f}s before next write")
                await asyncio.sleep(wait_time)
            
            input_ids_array = np.concatenate(self.accumulated_input_ids)
            steps_array = np.array(self.accumulated_steps)
            table = self._create_parquet_table(input_ids_array, steps_array)
            
            start_time = time.time()
            
            async with aiofiles.open(str(filename), 'wb') as f:
                def write_func():
                    pq.write_table(
                        table,
                        str(filename),
                        compression='ZSTD',
                        compression_level=self.compression_level,
                        row_group_size=min(len(steps_array), 10000)
                    )
                
                await self.loop.run_in_executor(self.executor, write_func)
            
            duration = time.time() - start_time
            self.last_write_time = time.time()
            
            # Get file size
            async with aiofiles.open(filename, 'rb') as f:
                file_size = await f.seek(0, 2)
            
            write_speed = file_size / (1024 * 1024 * duration)
            self.total_files_written += 1
            self.write_times.append(duration)
            
            logger.info(
                f"Wrote {filename.name}: "
                f"{len(steps_array)} samples, "
                f"{file_size / (1024*1024):.1f}MB, "
                f"{write_speed:.1f}MB/s, "
                    f"{duration:.2f}s"
                )
        except Exception as e:
            logger.error(f"Error writing file {filename}: {str(e)}", exc_info=True)
            raise

    def log_status(self):
        """Log current status of the data saver"""
        logger.info(
            f"DataSaver Status:\n"
            f"- Current file ID: {self.current_file_id}\n"
            f"- Samples in current file: {self.samples_in_current_file}\n"
            f"- Total samples processed: {self.total_samples_processed}\n"
            f"- Queue size: {self.batch_queue.qsize()}\n"
            f"- Processing task active: {self.processing_task and not self.processing_task.done()}\n"
            f"- Current checkpoint: {self.current_checkpoint}"
        )

    async def _enqueue_batch(self, state: State, batch: Dict[str, torch.Tensor]):
        """Enqueue a batch for processing."""
        if "input_ids" not in batch:
            return
        await self.batch_queue.put((state, batch))

    async def _process_queue(self):
        """Process batches strictly sequentially."""
        while True:
            try:
                state, batch = await self.batch_queue.get()
                await self._process_batch(state, batch)
                self.batch_queue.task_done()
            except asyncio.CancelledError:
                logger.info("Processing task cancelled")
                break
            except Exception as e:
                logger.error(f"Error processing batch: {str(e)}", exc_info=True)
                raise

   

    def _create_parquet_table(self, input_ids_array: np.ndarray, steps_array: np.ndarray) -> pa.Table:
        """Create PyArrow table from numpy arrays."""
        steps = pa.array(steps_array.astype(np.uint64))
        input_ids_list = [row.tolist() for row in input_ids_array]
        input_ids = pa.array(input_ids_list, type=pa.list_(pa.uint16()))
        
        return pa.Table.from_arrays(
            [input_ids, steps],
            names=['input_ids', 'step'],
            metadata={
                b'sequence_length': str(input_ids_array.shape[1]).encode()
            }
        )

    async def _handle_checkpoint(self, state: State) -> None:
        """Handle checkpoint transitions."""
        if self.last_checkpoint_batch != state.timestamp.batch:
            logger.info(f"Creating checkpoint {self.current_checkpoint + 1}")
            
            # Wait for queue to be empty
            await self.batch_queue.join()
            
            # Write any remaining data
            if self.accumulated_steps:
                save_folder = Path(state.model.config.data_save_folder)
                await self._write_file(save_folder)
                self.accumulated_input_ids = []
                self.accumulated_steps = []
                self.samples_in_current_file = 0
            
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
            self.loop.run_until_complete(self._handle_checkpoint(state))
            
    def iteration_checkpoint(self, state: State, logger: Logger) -> None:
        if self.save_interval(state, Event.ITERATION_CHECKPOINT):
            self.loop.run_until_complete(self._handle_checkpoint(state))
        
    def epoch_checkpoint(self, state: State, logger: Logger) -> None:
        if self.save_interval(state, Event.EPOCH_CHECKPOINT):
            self.loop.run_until_complete(self._handle_checkpoint(state))
    
    def close(self, state: Optional[State] = None, logger: Optional[Logger] = None) -> None:
        """Clean up and log statistics."""
        if logger is None:
            logger = logging.getLogger(__name__)
        logger.info("Shutting down AsyncOptimizedDataSaver...")
        
        try:
            # Cancel the processing task if it's running
            if self.processing_task and not self.processing_task.done():
                self.processing_task.cancel()
                self.loop.run_until_complete(self.processing_task)
            
            # Write any remaining data
            if state and self.accumulated_steps:
                save_folder = Path(state.model.config.data_save_folder)
                self.loop.run_until_complete(self._write_file(save_folder))
            
            # Log final statistics
            elapsed = time.time() - self.start_time
            avg_write_time = np.mean(self.write_times) if self.write_times else 0
            logger.info(
                f"AsyncOptimizedDataSaver finished:\n"
                f"- Total samples processed: {self.total_samples_processed}\n"
                f"- Total files written: {self.total_files_written}\n"
                f"- Average write time: {avg_write_time:.2f}s\n"
                f"- Average processing rate: {self.total_samples_processed / elapsed:.1f} samples/s\n"
                f"- Total runtime: {elapsed:.1f}s"
            )
            
        except Exception as e:
            logger.error("Error during shutdown:", exc_info=True)
            raise
        finally:
            self.executor.shutdown(wait=True)

    def __del__(self):
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
        if hasattr(self, 'loop') and self.loop:
            try:
                self.loop.close()
            except:
                pass