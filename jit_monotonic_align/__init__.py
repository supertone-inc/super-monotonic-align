import torch


@torch.no_grad()
@torch.jit.script
def maximum_path1(logp: torch.Tensor, attn_mask: torch.Tensor):
    # logp: [B, Tx, Ty], attn_mask: [B, Tx, Ty]
    B, Tx, Ty = logp.size()
    device = logp.device
    logp = logp * attn_mask  # [B, Tx, Ty]
    path = torch.zeros_like(logp)  # [B, Tx, Ty]
    max_neg_val = torch.tensor(-1e9, dtype=logp.dtype, device=device)

    x_len = attn_mask[:, :, 0].sum(dim=1).long()  # [B]
    y_len = attn_mask[:, 0, :].sum(dim=1).long()  # [B]

    for b in range(B):
        path[b, x_len[b] - 1, y_len[b] - 1] = 1

    # logp to cumulative logp
    logp[:, 1:, 0] = max_neg_val

    for ty in range(1, Ty):
        logp_prev_frame_1 = logp[:, :, ty - 1]  # [B, Tx]
        logp_prev_frame_2 = torch.roll(logp_prev_frame_1, shifts=1, dims=1)  # [B, Tx]
        logp_prev_frame_2[:, 0] = max_neg_val
        logp_prev_frame_max = torch.where(logp_prev_frame_1 > logp_prev_frame_2, logp_prev_frame_1, logp_prev_frame_2)
        logp[:, :, ty] += logp_prev_frame_max

    ids = torch.ones_like(x_len, device=device) * (x_len - 1)  # [B]
    arange = torch.arange(B, device=device)
    path = path.permute(2, 0, 1).contiguous()  # [Ty, B, Tx]
    attn_mask = attn_mask.permute(2, 0, 1).contiguous()  # [Ty, B, Tx]
    y_len_minus_1 = y_len - 1  # [B]
    for ty in range(Ty - 1, 0, -1):
        logp_prev_frame_1 = logp[:, :, ty - 1]  # [B, Tx]
        logp_prev_frame_2 = torch.roll(logp_prev_frame_1, shifts=1, dims=1)  # [B, Tx]
        logp_prev_frame_2[:, 0] = max_neg_val
        direction = torch.where(logp_prev_frame_1 > logp_prev_frame_2, 0, -1)  # [B, Tx]
        gathered_dir = torch.gather(direction, 1, ids.view(-1, 1)).view(-1)  # [B]
        gathered_dir.masked_fill_(ty > y_len_minus_1, 0)
        ids.add_(gathered_dir)
        path[ty - 1, arange, ids] = 1
    path *= attn_mask
    path = path.permute(1, 2, 0)  # [B, Tx, Ty]
    return path


@torch.no_grad()
def maximum_path2(logp: torch.Tensor, attn_mask: torch.Tensor):
    @torch.jit.script
    def cumulative_logp(logp, attn_mask):
        B, Tx, Ty = logp.size()
        device = logp.device
        logp = logp * attn_mask  # [B, Tx, Ty]
        path = torch.zeros_like(logp)  # [B, Tx, Ty]
        max_neg_val = torch.tensor(-1e9, dtype=logp.dtype, device=device)

        x_len = attn_mask[:, :, 0].sum(dim=1).long()  # [B]
        y_len = attn_mask[:, 0, :].sum(dim=1).long()  # [B]

        for b in range(B):
            path[b, x_len[b] - 1, y_len[b] - 1] = 1

        # logp to cumulative logp
        logp[:, 1:, 0] = max_neg_val

        for ty in range(1, Ty):
            logp_prev_frame_1 = logp[:, :, ty - 1]  # [B, Tx]
            logp_prev_frame_2 = torch.roll(logp_prev_frame_1, shifts=1, dims=1)  # [B, Tx]
            logp_prev_frame_2[:, 0] = max_neg_val
            logp_prev_frame_max = torch.where(
                logp_prev_frame_1 > logp_prev_frame_2, logp_prev_frame_1, logp_prev_frame_2
            )
            logp[:, :, ty] += logp_prev_frame_max
        return logp, x_len, y_len, path

    device = logp.device
    logp, x_len, y_len, path = cumulative_logp(logp, attn_mask)
    B, Tx, Ty = logp.size()
    logp = logp.detach().cpu().numpy()
    x_len = x_len.detach().cpu().numpy()
    y_len = y_len.detach().cpu().numpy()
    path = path.detach().cpu().numpy()
    # backtracking (naive)
    for b in range(B):
        idx = x_len[b] - 1
        path[b, x_len[b] - 1, y_len[b] - 1] = 1
        for ty in range(y_len[b] - 1, 0, -1):
            if idx != 0 and logp[b, idx - 1, ty - 1] > logp[b, idx, ty - 1]:
                idx = idx - 1
            path[b, idx, ty - 1] = 1
    path = torch.from_numpy(path).to(device)
    return path

