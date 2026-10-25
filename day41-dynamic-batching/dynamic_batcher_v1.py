
import queue
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor

import torch
from torch import nn


class DynamicBatcher:
    def __init__(self, model, max_batch_size=8, max_wait_ms=10):
        self.model = model.eval()
        self.device = next(model.parameters()).device
        self.max_batch_size = max_batch_size
        self.max_wait = max_wait_ms / 1000.0

        self.requests = queue.Queue()
        self.worker = threading.Thread(
            target=self._process_batches,
            daemon=True
        )
        self.worker.start()

    def submit(self, tensor):
        future = Future()
        arrival = time.perf_counter()
        self.requests.put((arrival, tensor, future))
        return future

    def _process_batches(self):
        while True:
            first = self.requests.get()
            batch = [first]

            # Deadline starts with the oldest request.
            deadline = first[0] + self.max_wait

            # while len(batch) < self.max_batch_size:
            #     remaining = deadline - time.perf_counter()

            #     if remaining <= 0:
            #         break

            #     try:
            #         item = self.requests.get(timeout=remaining)
            #         batch.append(item)
            #     except queue.Empty:
            #         break
            # buggy - It fails to check if we have any requests queued before dispatching when the initial deadline has passed
            
            while len(batch) < self.max_batch_size:

                # First: collect requests already available.
                try:
                    item = self.requests.get_nowait()
                    batch.append(item)
                    continue
                except queue.Empty:
                    pass

                # Only wait if the queue is currently empty.
                remaining = deadline - time.perf_counter()

                if remaining <= 0:
                    break

                try:
                    item = self.requests.get(timeout=remaining)
                    batch.append(item)
                except queue.Empty:
                    break


            try:
                inputs = torch.stack(
                    [item[1] for item in batch]
                ).to(self.device)

                start = time.perf_counter()

                with torch.inference_mode():
                    outputs = self.model(inputs)

                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)

                elapsed_ms = (
                    time.perf_counter() - start
                ) * 1000

                for item, output in zip(batch, outputs):
                    item[2].set_result(output.detach().cpu())

                print(
                    f"Batch size={len(batch):2d} | "
                    f"Inference={elapsed_ms:.3f} ms"
                )

            except Exception as exc:
                for item in batch:
                    item[2].set_exception(exc)


if __name__ == "__main__":
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    ).to(device)

    batcher = DynamicBatcher(
        model,
        max_batch_size=8,
        max_wait_ms=10
    )

    # Simulate 32 independently submitted requests.
    def send_request(i):
        return batcher.submit(torch.randn(128))

    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = list(pool.map(send_request, range(32)))

    results = [f.result(timeout=30) for f in futures]

    print(f"Completed requests: {len(results)}")
    print(f"Output shape: {results[0].shape}")
