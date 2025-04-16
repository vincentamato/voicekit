import torch
import torch.nn as nn

class LengthRegulator(nn.Module):
    """
    LengthRegulator expands phoneme-level embeddings to frame-level by repeating
    each time-step according to the provided durations (in frames).
    """
    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        durations: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, dim)
            durations: LongTensor of shape (batch, seq_len), durations in frames
        Returns:
            Tensor of shape (batch, max_duration_sum, dim), padded
        """
        batch_size, seq_len, dim = x.size()
        outputs = []
        max_len = 0
        # Repeat each embedding by its duration
        for b in range(batch_size):
            reps = durations[b].tolist()
            expanded = x[b].repeat_interleave(torch.tensor(reps, dtype=torch.long), dim=0)
            outputs.append(expanded)
            if expanded.size(0) > max_len:
                max_len = expanded.size(0)
        # Pad sequences to max_len
        padded = x.new_zeros(batch_size, max_len, dim)
        for b, seq in enumerate(outputs):
            length = seq.size(0)
            padded[b, :length, :] = seq
        return padded
