"""Minimal two-GPU FSDP smoke test for the yyl training host."""

import torch
from accelerate import Accelerator
from safetensors.torch import save_file
from torch.utils.data import DataLoader, TensorDataset
from torch import nn


def main() -> None:
    accelerator = Accelerator()
    if accelerator.num_processes != 2:
        raise RuntimeError(f"Expected 2 FSDP ranks, got {accelerator.num_processes}.")

    model = nn.Sequential(
        nn.Linear(2048, 2048),
        nn.GELU(),
        nn.Linear(2048, 2048),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    dataset = TensorDataset(torch.randn(64, 2048))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    batch = next(iter(dataloader))[0]
    local_size = torch.tensor([batch.shape[0]], device=accelerator.device)
    global_sizes = accelerator.gather(local_size)
    if global_sizes.sum().item() != 64 or not torch.all(global_sizes == 32):
        raise RuntimeError(f"Expected two local batches of 32, got {global_sizes.tolist()}")
    loss = model(batch).square().mean()
    accelerator.backward(loss)
    optimizer.step()
    optimizer.zero_grad()
    accelerator.wait_for_everyone()
    # FSDP gathers a full state dict collectively: every rank must call this.
    state_dict = accelerator.get_state_dict(model)

    if accelerator.is_main_process:
        save_file(state_dict, "/tmp/yyl_fsdp_smoke.safetensors")
        print(
            f"FSDP_SMOKE_OK processes={accelerator.num_processes} "
            f"global_batch={global_sizes.sum().item()} local_batches={global_sizes.tolist()} "
            f"device={accelerator.device} loss={loss.item():.6f} checkpoint=/tmp/yyl_fsdp_smoke.safetensors",
            flush=True,
        )


if __name__ == "__main__":
    main()
