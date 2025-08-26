import torch
import numpy as np
import pdb

debug_cnt = -1

def make_batch(augmentor, difficulty = 0.3, train = True):
    Hs = []
    img_list = augmentor.train if train else augmentor.test
    dev = augmentor.device
    batch_images = []

    with torch.no_grad(): # we dont require grads in the augmentation
        for b in range(augmentor.batch_size):
            rdidx = np.random.randint(len(img_list))
            img = torch.tensor(img_list[rdidx], dtype=torch.float32).permute(2,0,1).to(augmentor.device).unsqueeze(0)
            batch_images.append(img)

        batch_images = torch.cat(batch_images)

        p1, H1 = augmentor(batch_images, difficulty)
        p2, H2 = augmentor(batch_images, difficulty, TPS = True, prob_deformation = 0.7)

    return p1, p2, H1, H2


def plot_corrs(p1, p2, src_pts, tgt_pts):
    import matplotlib.pyplot as plt
    p1 = p1.cpu()
    p2 = p2.cpu()
    src_pts = src_pts.cpu() ; tgt_pts = tgt_pts.cpu()
    rnd_idx = np.random.randint(len(src_pts), size=200)
    src_pts = src_pts[rnd_idx, ...]
    tgt_pts = tgt_pts[rnd_idx, ...]

    #Plot ground-truth correspondences
    fig, ax = plt.subplots(1,2,figsize=(18, 12))
    colors = np.random.uniform(size=(len(tgt_pts),3))
    #Src image
    img = p1
    for i, p in enumerate(src_pts):
        ax[0].scatter(p[0],p[1],color=colors[i])
    ax[0].imshow(img.permute(1,2,0).numpy()[...,::-1])

    #Target img
    img2 = p2
    for i, p in enumerate(tgt_pts):
        ax[1].scatter(p[0],p[1],color=colors[i])
    ax[1].imshow(img2.permute(1,2,0).numpy()[...,::-1])
    plt.show()


# def get_corresponding_pts(p1, p2, H, H2, augmentor, h, w, crop = None):
#     '''
#         Get dense corresponding points
#     '''
#     global debug_cnt
#     negatives, positives = [], []
#
#     with torch.no_grad():
#         #real input res of samples
#         rh, rw = p1.shape[-2:]
#         ratio = torch.tensor([rw/w, rh/h], device = p1.device)
#
#         (H, mask1) = H
#         (H2, src, W, A, mask2) = H2
#
#         #Generate meshgrid of target pts
#         x, y = torch.meshgrid(torch.arange(w, device=p1.device), torch.arange(h, device=p1.device), indexing ='xy')
#         mesh = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1)], dim=-1)
#         target_pts = mesh.view(-1, 2) * ratio
#
#         #Pack all transformations into T
#         for batch_idx in range(len(p1)):
#             with torch.no_grad():
#                 T = (H[batch_idx], H2[batch_idx],
#                     src[batch_idx].unsqueeze(0), W[batch_idx].unsqueeze(0), A[batch_idx].unsqueeze(0))
#                 #We now warp the target points to src image
#                 src_pts = (augmentor.get_correspondences(target_pts, T) ) #target to src
#                 tgt_pts = (target_pts)
#
#                 #Check out of bounds points
#                 mask_valid = (src_pts[:, 0] >=0) & (src_pts[:, 1] >=0) & \
#                             (src_pts[:, 0] < rw) & (src_pts[:, 1] < rh)
#
#                 negatives.append( tgt_pts[~mask_valid] )
#                 tgt_pts = tgt_pts[mask_valid]
#                 src_pts = src_pts[mask_valid]
#
#
#                 #Remove invalid pixels
#                 mask_valid =    mask1[batch_idx, src_pts[:,1].long(), src_pts[:,0].long()]  & \
#                                 mask2[batch_idx, tgt_pts[:,1].long(), tgt_pts[:,0].long()]
#                 tgt_pts = tgt_pts[mask_valid]
#                 src_pts = src_pts[mask_valid]
#
#                 # limit nb of matches if desired
#                 if crop is not None:
#                     rnd_idx = torch.randperm(len(src_pts), device=src_pts.device)[:crop]
#                     src_pts = src_pts[rnd_idx]
#                     tgt_pts = tgt_pts[rnd_idx]
#
#                 if debug_cnt >=0 and debug_cnt < 4:
#                     plot_corrs(p1[batch_idx], p2[batch_idx], src_pts , tgt_pts )
#                     debug_cnt +=1
#
#                 src_pts = (src_pts / ratio)
#                 tgt_pts = (tgt_pts / ratio)
#
#                 #Check out of bounds points
#                 padto = 10 if crop is not None else 2
#                 mask_valid1 = (src_pts[:, 0] >= (0 + padto)) & (src_pts[:, 1] >= (0 + padto)) & \
#                              (src_pts[:, 0] < (w - padto)) & (src_pts[:, 1] < (h - padto))
#                 mask_valid2 = (tgt_pts[:, 0] >= (0 + padto)) & (tgt_pts[:, 1] >= (0 + padto)) & \
#                              (tgt_pts[:, 0] < (w - padto)) & (tgt_pts[:, 1] < (h - padto))
#                 mask_valid = mask_valid1 & mask_valid2
#                 tgt_pts = tgt_pts[mask_valid]
#                 src_pts = src_pts[mask_valid]
#
#                 #Remove repeated correspondences
#                 lut_mat = torch.ones((h, w, 4), device = src_pts.device, dtype = src_pts.dtype) * -1
#                 # src_pts_np = src_pts.cpu().numpy()
#                 # tgt_pts_np = tgt_pts.cpu().numpy()
#                 try:
#                     lut_mat[src_pts[:,1].long(), src_pts[:,0].long()] = torch.cat([src_pts, tgt_pts], dim=1)
#                     mask_valid = torch.all(lut_mat >= 0, dim=-1)
#                     points = lut_mat[mask_valid]
#                     positives.append(points)
#                 except:
#                     pdb.set_trace()
#                     print('..')
#
#     return negatives, positives

def get_corresponding_pts(p1, p2, H, H2, augmentor, h, w, crop=None):
    '''
        Get dense corresponding points
    '''
    global debug_cnt
    negatives, positives = [], []

    with torch.no_grad():
        # real input res of samples
        rh, rw = p1.shape[-2:]
        ratio = torch.tensor([rw / w, rh / h], device=p1.device)

        (H, mask1) = H
        (H2, src, W, A, mask2) = H2

        # Generate meshgrid of target pts
        x, y = torch.meshgrid(torch.arange(w, device=p1.device), torch.arange(h, device=p1.device), indexing='xy')
        mesh = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1)], dim=-1)
        target_pts = mesh.view(-1, 2) * ratio

        # Pack all transformations into T
        for batch_idx in range(len(p1)):
            with torch.no_grad():
                T = (H[batch_idx], H2[batch_idx],
                     src[batch_idx].unsqueeze(0), W[batch_idx].unsqueeze(0), A[batch_idx].unsqueeze(0))
                # We now warp the target points to src image
                src_pts = (augmentor.get_correspondences(target_pts, T))  # target to src
                tgt_pts = (target_pts)

                # Check out of bounds points
                mask_valid = (src_pts[:, 0] >= 0) & (src_pts[:, 1] >= 0) & \
                             (src_pts[:, 0] < rw) & (src_pts[:, 1] < rh)

                negatives.append(tgt_pts[~mask_valid])
                tgt_pts = tgt_pts[mask_valid]
                src_pts = src_pts[mask_valid]

                # Remove invalid pixels
                mask_valid = mask1[batch_idx, src_pts[:, 1].long(), src_pts[:, 0].long()] & \
                             mask2[batch_idx, tgt_pts[:, 1].long(), tgt_pts[:, 0].long()]
                tgt_pts = tgt_pts[mask_valid]
                src_pts = src_pts[mask_valid]

                # limit nb of matches if desired
                if crop is not None:
                    rnd_idx = torch.randperm(len(src_pts), device=src_pts.device)[:crop]
                    src_pts = src_pts[rnd_idx]
                    tgt_pts = tgt_pts[rnd_idx]

                if debug_cnt >= 0 and debug_cnt < 4:
                    plot_corrs(p1[batch_idx], p2[batch_idx], src_pts, tgt_pts)
                    debug_cnt += 1

                src_pts = (src_pts / ratio)
                tgt_pts = (tgt_pts / ratio)

                # Check out of bounds points
                padto = 10 if crop is not None else 2
                mask_valid1 = (src_pts[:, 0] >= (0 + padto)) & (src_pts[:, 1] >= (0 + padto)) & \
                              (src_pts[:, 0] < (w - padto)) & (src_pts[:, 1] < (h - padto))
                mask_valid2 = (tgt_pts[:, 0] >= (0 + padto)) & (tgt_pts[:, 1] >= (0 + padto)) & \
                              (tgt_pts[:, 0] < (w - padto)) & (tgt_pts[:, 1] < (h - padto))
                mask_valid = mask_valid1 & mask_valid2
                tgt_pts = tgt_pts[mask_valid]
                src_pts = src_pts[mask_valid]

                # Remove repeated correspondences
                lut_mat = torch.ones((h, w, 4), device=src_pts.device, dtype=src_pts.dtype) * -1
                # src_pts_np = src_pts.cpu().numpy()
                # tgt_pts_np = tgt_pts.cpu().numpy()
                try:
                    lut_mat[src_pts[:, 1].long(), src_pts[:, 0].long()] = torch.cat([src_pts, tgt_pts], dim=1)
                    mask_valid = torch.all(lut_mat >= 0, dim=-1)
                    points = lut_mat[mask_valid]
                    positives.append(points)
                except:
                    pdb.set_trace()
                    print('..')

    return negatives, positives


# def get_corresponding_pts(p1, p2, H, H2, augmentor, h, w, crop=None):
#     '''
#         获取稠密对应点（四个角点的坐标）
#         参数：
#         - p1, p2: 原图像和目标图像特征图
#         - H: 源图到目标图的单应性矩阵
#         - H2: 目标图到源图的单应性矩阵
#         - augmentor: 用于坐标变换
#         - h, w: 网格分辨率
#     '''
#     negatives, positives = [], []
#
#     with torch.no_grad():
#         # 实际分辨率
#         rh, rw = p1.shape[-2:]  # p1 的实际高度和宽度
#         ratio = torch.tensor([rw / w, rh / h], device=p1.device)  # 源图像到目标图像的比例
#
#         # 单应性矩阵解包
#         (H, mask1) = H
#         (H2, src, W, A, mask2) = H2
#
#         # 网格大小（假设固定为 7）
#         grid_size = 7
#         x, y = torch.meshgrid(
#             torch.arange(w, device=p1.device),
#             torch.arange(h, device=p1.device),
#             indexing='xy'
#         )
#         mesh = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1)], dim=-1)  # (h, w, 2)
#         mesh = mesh.view(-1, 2) * ratio  # 将网格点坐标调整到实际分辨率
#
#         # 为每个网格点生成四个角点
#         corner_offsets = torch.tensor(
#             [[0, 0], [grid_size, 0], [0, grid_size], [grid_size, grid_size]],
#             device=p1.device
#         )
#         corner_points = (mesh.unsqueeze(1) + corner_offsets).view(-1, 2)  # (N * 4, 2)
#         target_pts = corner_points
#
#         for batch_idx in range(len(p1)):
#             with torch.no_grad():
#                 # 打包当前批次的单应性矩阵和相关参数
#                 T = (H[batch_idx], H2[batch_idx],
#                      src[batch_idx].unsqueeze(0),
#                      W[batch_idx].unsqueeze(0),
#                      A[batch_idx].unsqueeze(0))
#
#                 # 将目标图像的角点变换到源图像中
#                 src_corner_points = augmentor.get_correspondences(target_pts, T)  # (N * 4, 2)
#                 tgt_corner_points = target_pts  # 目标图像的角点
#
#                 # 检查源图像中的点是否有效（是否在图像范围内）
#                 valid_mask = (src_corner_points[:, 0] >= 0) & (src_corner_points[:, 1] >= 0) & \
#                              (src_corner_points[:, 0] < rw) & (src_corner_points[:, 1] < rh)
#
#                 # 对所有角点进行分组，确保每个网格的 4 个角点都有效
#                 num_corners = 4
#                 valid_mask_grouped = valid_mask.view(-1, num_corners).all(dim=1)  # 每个网格的有效性
#
#                 # 仅保留有效的角点
#                 src_corner_points = src_corner_points.view(-1, num_corners, 2)[valid_mask_grouped]
#                 tgt_corner_points = tgt_corner_points.view(-1, num_corners, 2)[valid_mask_grouped]
#
#                 # 按比例调整回目标图像的分辨率
#                 src_corner_points = src_corner_points / ratio
#                 tgt_corner_points = tgt_corner_points / ratio
#
#                 # 拼接源图像和目标图像的角点
#                 valid_corrs = torch.cat([src_corner_points, tgt_corner_points], dim=-1)
#
#                 # 如果有有效匹配点，保存到批次结果
#                 if valid_corrs.size(0) > 0:
#                     positives.append(valid_corrs)
#
#     return negatives, positives
#
#
#
# #batchsize,N,4,4
#

def crop_patches(tensor, coords, size = 7):
    '''
        Crop [size x size] patches around 2D coordinates from a tensor.
    '''
    B, C, H, W = tensor.shape

    x, y = coords[:, 0], coords[:, 1]
    y = y.view(-1, 1, 1)
    x = x.view(-1, 1, 1)
    halfsize = size // 2
    # Create meshgrid for indexing
    x_offset, y_offset = torch.meshgrid(torch.arange(-halfsize, halfsize+1), torch.arange(-halfsize, halfsize+1), indexing='xy')
    y_offset = y_offset.to(tensor.device)
    x_offset = x_offset.to(tensor.device)

    # Compute indices around each coordinate
    y_indices = (y + y_offset.view(1, size, size)).squeeze(0) + halfsize
    x_indices = (x + x_offset.view(1, size, size)).squeeze(0) + halfsize

    # Handle out-of-boundary indices with padding
    tensor_padded = torch.nn.functional.pad(tensor, (halfsize, halfsize, halfsize, halfsize), mode='constant')

    # Index tensor to get patches
    patches = tensor_padded[:, :, y_indices, x_indices] # [B, C, N, H, W]
    return patches

def subpix_softmax2d(heatmaps, temp = 0.25):
    N, H, W = heatmaps.shape
    heatmaps = torch.softmax(temp * heatmaps.view(-1, H*W), -1).view(-1, H, W)
    x, y = torch.meshgrid(torch.arange(W, device =  heatmaps.device ), torch.arange(H, device =  heatmaps.device ), indexing = 'xy')
    x = x - (W//2)
    y = y - (H//2)
    #pdb.set_trace()
    coords_x = (x[None, ...] * heatmaps)
    coords_y = (y[None, ...] * heatmaps)
    coords = torch.cat([coords_x[..., None], coords_y[..., None]], -1).view(N, H*W, 2)
    coords = coords.sum(1)

    return coords


def check_accuracy(X, Y, pts1 = None, pts2 = None, plot=False):
    with torch.no_grad():
        #dist_mat = torch.cdist(X,Y)
        dist_mat = X @ Y.t()
        nn = torch.argmax(dist_mat, dim=1)
        #nn = torch.argmin(dist_mat, dim=1)
        correct = nn == torch.arange(len(X), device = X.device)

        if pts1 is not None and plot:
            import matplotlib.pyplot as plt
            canvas = torch.zeros((60, 80),device=X.device)
            pts1 = pts1[~correct]
            canvas[pts1[:,1].long(), pts1[:,0].long()] = 1
            canvas = canvas.cpu().numpy()
            plt.imshow(canvas), plt.show()

        acc = correct.sum().item() / len(X)
        return acc

def get_nb_trainable_params(model):
	model_parameters = filter(lambda p: p.requires_grad, model.parameters())
	nb_params = sum([np.prod(p.size()) for p in model_parameters])
 
	print('Number of trainable parameters: {:d}'.format(nb_params))