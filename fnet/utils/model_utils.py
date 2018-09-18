import torch


def move_optim(optim: torch.optim.Optimizer, device: torch.device):
    """Moves optimizer parameters to specified device.

    """
    for g_state in optim.state.values():
        for k, v in g_state.items():
            if torch.is_tensor(v):
                g_state[k] = v.to(device)
