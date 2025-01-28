import composer
import composer.utils.dist as dist
import torch
import time


if __name__ == '__main__':
    print("Barrier script started")
    device = composer.utils.get_device()
    composer.utils.dist.initialize_dist(device, 10)
    sleep_time = time.sleep(dist.get_global_rank())
    print(f"[GR {dist.get_global_rank()}] waiting")
    composer.utils.dist.barrier()
    torch.distributed.destroy_process_group()
    print("Barrier script Done")


