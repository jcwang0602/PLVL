import torch
import torch.nn.functional as F
from torchvision.utils import save_image


def vis_tokens(img, idx_token, edge_color=[1.0, 1.0, 1.0], edge_width=1):
    """Visualize tokens
    Return:
        vis_img (Tensor[B, 3, H, W]): visualize result.

    Args:
        img (Tensor[B, 3, H, W]): input image.
        token_dict (dict): dict for input token information
        edge_color (float[int]): color for edges
        edge_width (int): width for edges
    """

    N = len(torch.unique(idx_token))
    device, dtype = img.device, img.dtype

    # color_map = torch.tensor(img, device=device, dtype=float) / 255.0
    # color_map = color_map.permute(2, 0, 1)[None, ...]
    color_map = F.avg_pool2d(img, kernel_size=16) # [1,3,28,28]
    
    B, C, H, W = color_map.shape

    token_color = map2token(color_map, idx_token)
    vis_img = token2map(token_color, idx_token)

    token_idx = torch.arange(N, device=device)[None, :, None].float() / N
    idx_map = token2map(token_idx, idx_token)  # [B, 1, H, W]

    vis_img = F.interpolate(vis_img, [H * 16, W * 16], mode='nearest')
    idx_map = F.interpolate(idx_map, [H * 16, W * 16], mode='nearest')

    kernel = idx_map.new_zeros([4, 1, 3, 3])
    kernel[:, :, 1, 1] = 1
    kernel[0, :, 0, 1] = -1
    kernel[1, :, 2, 1] = -1
    kernel[2, :, 1, 0] = -1
    kernel[3, :, 1, 2] = -1

    for i in range(edge_width):
        edge_map = F.conv2d(F.pad(idx_map, [1, 1, 1, 1], mode='replicate'), kernel)
        edge_map = (edge_map != 0).max(dim=1, keepdim=True)[0]
        idx_map = idx_map * (~edge_map) + torch.rand(idx_map.shape, device=device, dtype=dtype) * edge_map
    edge_color = torch.tensor(edge_color, device=device, dtype=dtype)[None, :, None, None]
    vis_img = vis_img * (~edge_map) + edge_color * edge_map
    return vis_img


def map2token(feature_map, idx_token):
    """Transform feature map to vision tokens. This function only
    works when the resolution of the feature map is not higher than
    the initial grid structure.

    Returns:
        out (Tensor[B, N, C]): token features.

    Args:
        feature_map (Tensor[B, C, H, W]): feature map.
        token_dict (dict): dict for token information.
    """
    N = len(torch.unique(idx_token))
    H_init, W_init = 28,28
    N_init = H_init * W_init

    # agg_weight = token_dict['agg_weight'] if 'agg_weight' in token_dict.keys() else None
    agg_weight = None  # we do not use the weight value here

    B, C, H, W = feature_map.shape
    device = feature_map.device

    if N_init == N and N == H * W:
        # for the initial tokens with grid structure, just reshape
        return feature_map.flatten(2).permute(0, 2, 1).contiguous()

    idx_hw = get_grid_index([H_init, W_init], [H, W], device=device)[None, :].expand(B, -1)

    idx_batch = torch.arange(B, device=device)[:, None].expand(B, N_init)
    if agg_weight is None:
        value = feature_map.new_ones(B * N_init)
    else:
        value = agg_weight.reshape(B * N_init).type(feature_map.dtype)

    # choose the way with fewer flops.
    if N_init < N * H * W:
        # use sparse matrix multiplication
        # Flops: B * N_init * (C+2)
        idx_token = idx_token + idx_batch * N
        idx_hw = idx_hw + idx_batch * H * W
        indices = torch.stack([idx_token, idx_hw], dim=0).reshape(2, -1)

        # torch.sparse do not support fp16
        with torch.cuda.amp.autocast(enabled=False):
            # sparse mm do not support gradient for sparse matrix
            value = value.detach().float()
            # build a sparse matrix with shape [B*N, B*H*W]
            A = torch.sparse_coo_tensor(indices, value, (B * N, B * H * W))
            # normalize the matrix
            all_weight = A @ torch.ones(
                [B * H * W, 1], device=device, dtype=torch.float32) + 1e-6
            value = value / all_weight[idx_token.reshape(-1), 0]

            A = torch.sparse_coo_tensor(indices, value, (B * N, B * H * W))
            # out: [B*N, C]
            out = A @ feature_map. \
                permute(0, 2, 3, 1).contiguous().reshape(B * H * W, C).float()
    else:
        # use dense matrix multiplication
        # Flops: B * N * H * W * (C+2)
        indices = torch.stack([idx_batch, idx_token, idx_hw], dim=0).reshape(3, -1)
        value = value.detach()  # To reduce the training time, we detach here.
        A = torch.sparse_coo_tensor(indices, value, (B, N, H * W)).to_dense()
        # normalize the matrix
        A = A / (A.sum(dim=-1, keepdim=True) + 1e-6)

        out = A @ feature_map.permute(0, 2, 3, 1).reshape(B, H * W, C).contiguous()

    out = out.type(feature_map.dtype)
    out = out.reshape(B, N, C)
    return out

def token2map(x, idx_token):
    """Transform vision tokens to feature map. This function only
    works when the resolution of the feature map is not higher than
    the initial grid structure.
    Returns:
        x_out (Tensor[B, C, H, W]): feature map.

    Args:
        token_dict (dict): dict for token information.
    """

    H, W = 28, 28 # 图片呢聚类之前的大小
    H_init, W_init = 28, 28 # 图片ptach之后的大小
    B, N, C = x.shape # 聚类之后的Token序列 
    N_init = H_init * W_init # 初始的序列长度
    device = x.device

    # 如果初始的Token序列长度=现在的Token序列长度 并且现在的序列长度=当前map的长度 
    if N_init == N and N == H * W:
        # for the initial tokens with grid structure, just reshape
        return x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

    # 对于每一个init的Token，获得其对应 flattened 之后的索引   应该是1-28*28
    # for each initial grid, get the corresponding index in the flattened feature map.
    idx_hw = get_grid_index([H_init, W_init], [H, W], device=device)[None, :].expand(B, -1)
    # 获得整个batch的索引
    idx_batch = torch.arange(B, device=device)[:, None].expand(B, N_init)
    # 获得整个batchs的 1， 这个也会是最终的一个完整的形状
    value = x.new_ones(B * N_init) # [784]

    # choose the way with fewer flops.
    if N_init < N * H * W:
        # use sparse matrix multiplication
        # Flops: B * N_init * (C+2)
        idx_hw = idx_hw + idx_batch * H * W
        idx_tokens = idx_token + idx_batch * N
        coor = torch.stack([idx_hw, idx_tokens], dim=0).reshape(2, B * N_init)

        # torch.sparse do not support fp16
        with torch.cuda.amp.autocast(enabled=False):
            # torch.sparse do not support gradient for
            # sparse tensor, so we detach it
            value = value.detach().float()
            # build a sparse matrix with the shape [B * H * W, B * N]
            A = torch.sparse_coo_tensor(coor, value, torch.Size([B * H * W, B * N])) # [784, 196]
            # normalize the weight for each row
            all_weight = A @ x.new_ones(B * N, 1).type(torch.float32) + 1e-6
            value = value / all_weight[idx_hw.reshape(-1), 0]
            # update the matrix with normalize weight
            A = torch.sparse_coo_tensor(coor, value, torch.Size([B * H * W, B * N]))
            # sparse matrix multiplication
            x_out = A @ x.reshape(B * N, C).type(torch.float32)  # [B*H*W, C]

    else:
        # use dense matrix multiplication
        # Flops: B * N * H * W * (C+2)
        coor = torch.stack([idx_batch, idx_hw, idx_token], dim=0).reshape(3, B * N_init)

        # build a matrix with shape [B, H*W, N]
        A = torch.sparse.FloatTensor(coor, value, torch.Size([B, H * W, N])).to_dense()
        # normalize the weight
        A = A / (A.sum(dim=-1, keepdim=True) + 1e-6)

        x_out = A @ x  # [B, H*W, C]

    x_out = x_out.type(x.dtype)
    x_out = x_out.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
    return x_out


def get_grid_index(init_size, map_size, device):
    """For each initial grid, get its index in the feature map.
    Returns:
        idx (LongTensor[B, N_init]): index in flattened feature map.

    Args:
        init_grid_size(list[int] or tuple[int]): initial grid resolution in
            format [H_init, W_init].
        map_size(list[int] or tuple[int]): feature map resolution in format
            [H, W].
        device: the device of output
    """
    H_init, W_init = init_size
    H, W = map_size
    idx = torch.arange(H * W, device=device).reshape(1, 1, H, W)
    idx = F.interpolate(idx.float(), [H_init, W_init], mode='nearest').long()
    return idx.flatten()
