
import os
import argparse

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend", choices=["gloo", "nccl"],
        default="gloo"
    )
    args = parser.parse_args()

    # torchrun provides these environment variables.
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if args.backend == "nccl":
        assert torch.cuda.is_available()
        assert local_rank < torch.cuda.device_count()
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    dist.init_process_group(backend=args.backend)

    try:
        print(
            f"Rank={rank}, World size={world_size}, "
            f"Device={device}", flush=True
        )

        # Synthetic dataset: y = 3x + 2
        x = torch.linspace(-1, 1, 128).unsqueeze(1)
        y = 3 * x + 2
        dataset = TensorDataset(x, y)

        # Give each rank a distinct portion of the data.
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )

        loader = DataLoader(
            dataset,
            batch_size=16,
            sampler=sampler
        )

        torch.manual_seed(42)

        # Every rank holds a complete model.
        model = nn.Linear(1, 1).to(device)

        if device.type == "cuda":
            model = DDP(model, device_ids=[local_rank])
        else:
            model = DDP(model)

        optimizer = torch.optim.SGD(
            model.parameters(), lr=0.15
        )
        criterion = nn.MSELoss()

        for epoch in range(10):
            sampler.set_epoch(epoch)
            total_loss = torch.zeros(1, device=device)

            if epoch == 0:
                print(
                    f"Rank {rank} first indices: "
                    f"{list(iter(sampler))[:6]}",
                    flush=True
                )

            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)

                optimizer.zero_grad()
                predictions = model(xb)
                loss = criterion(predictions, yb)

                loss.backward()  # DDP synchronizes gradients.
                optimizer.step()

                total_loss += loss.detach()

            # Aggregate loss for reporting.
            dist.all_reduce(
                total_loss, op=dist.ReduceOp.SUM
            )

            if rank == 0:
                mean_loss = (
                    total_loss.item()
                    / (len(loader) * world_size)
                )
                print(
                    f"Epoch {epoch + 1}: "
                    f"Mean loss = {mean_loss:.6f}"
                )

        # Compare final model parameters across ranks.
        parameters = torch.cat([
            p.detach().flatten()
            for p in model.module.parameters()
        ])

        gathered = [
            torch.zeros_like(parameters)
            for _ in range(world_size)
        ]
        dist.all_gather(gathered, parameters)

        if rank == 0:
            max_difference = max(
                (parameters - other).abs().max().item()
                for other in gathered
            )

            print("Final weight:",
                  model.module.weight.item())
            print("Final bias:",
                  model.module.bias.item())
            print("Max parameter difference:",
                  max_difference)

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
