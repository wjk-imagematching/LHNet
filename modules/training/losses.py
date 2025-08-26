import torch
import torch.nn.functional as F
import numpy as np
from modules.dataset.megadepth import megadepth_warper

from modules.training import utils

from third_party.alike_wrapper import extract_alike_kpts

def dual_softmax_loss(X, Y, temp = 0.2):
    if X.size() != Y.size() or X.dim() != 2 or Y.dim() != 2:
        raise RuntimeError('Error: X and Y shapes must match and be 2D matrices')

    dist_mat = (X @ Y.t()) * temp
    conf_matrix12 = F.log_softmax(dist_mat, dim=1)
    conf_matrix21 = F.log_softmax(dist_mat.t(), dim=1)

    with torch.no_grad():
        conf12 = torch.exp( conf_matrix12 ).max(dim=-1)[0]
        conf21 = torch.exp( conf_matrix21 ).max(dim=-1)[0]
        conf = conf12 * conf21

    target = torch.arange(len(X), device = X.device)

    loss = F.nll_loss(conf_matrix12, target) + \
           F.nll_loss(conf_matrix21, target)

    return loss, conf

def smooth_l1_loss(input, target, beta=2.0, size_average=True):
    diff = torch.abs(input - target)
    loss = torch.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
    return loss.mean() if size_average else loss.sum()

def fine_loss(f1, f2, pts1, pts2, fine_module, ws=7):
    '''
        Compute Fine features and spatial loss
    '''
    C, H, W = f1.shape
    N = len(pts1)

    #Sort random offsets
    with torch.no_grad():
        a = -(ws//2)
        b = (ws//2)
        offset_gt = (a - b) * torch.rand(N, 2, device = f1.device) + b
        pts2_random = pts2 + offset_gt

    #pdb.set_trace()
    patches1 = utils.crop_patches(f1.unsqueeze(0), (pts1+0.5).long(), size=ws).view(C, N, ws * ws).permute(1, 2, 0) #[N, ws*ws, C]
    patches2 = utils.crop_patches(f2.unsqueeze(0), (pts2_random+0.5).long(), size=ws).view(C, N, ws * ws).permute(1, 2, 0)  #[N, ws*ws, C]

    #Apply transformer
    patches1, patches2 = fine_module(patches1, patches2)

    features = patches1.view(N, ws, ws, C)[:, ws//2, ws//2, :].view(N, 1, 1, C) # [N, 1, 1, C]
    patches2 = patches2.view(N, ws, ws, C) # [N, w, w, C]

    #Dot Product
    heatmap_match = (features * patches2).sum(-1)
    offset_coords = utils.subpix_softmax2d(heatmap_match)

    #Invert offset because center crop inverts it
    offset_gt = -offset_gt 

    #MSE
    error = ((offset_coords - offset_gt)**2).sum(-1).mean()
    #error = smooth_l1_loss(offset_coords, offset_gt)
    return error


def alike_distill_loss(kpts, img):

    C, H, W = kpts.shape
    kpts = kpts.permute(1,2,0)
    img = img.permute(1,2,0).expand(-1,-1,3).cpu().numpy() * 255

    with torch.no_grad():
        alike_kpts = torch.tensor( extract_alike_kpts(img), device=kpts.device )
        labels = torch.ones((H, W), dtype = torch.long, device = kpts.device) * 64 # -> Default is non-keypoint (bin 64)
        offsets = (((alike_kpts/8) - (alike_kpts/8).long())*8).long()
        offsets =  offsets[:, 0] + 8*offsets[:, 1]  # Linear IDX
        labels[(alike_kpts[:,1]/8).long(), (alike_kpts[:,0]/8).long()] = offsets

    kpts = kpts.view(-1,C)
    labels = labels.view(-1)

    mask = labels < 64
    idxs_pos = mask.nonzero().flatten()
    idxs_neg = (~mask).nonzero().flatten()
    perm = torch.randperm(idxs_neg.size(0))[:len(idxs_pos)//32]
    idxs_neg = idxs_neg[perm]
    idxs = torch.cat([idxs_pos, idxs_neg])

    kpts = kpts[idxs]
    labels = labels[idxs]

    with torch.no_grad():
        predicted = kpts.max(dim=-1)[1]
        acc =  (labels == predicted)
        acc = acc.sum() / len(acc)

    kpts = F.log_softmax(kpts)
    loss = F.nll_loss(kpts, labels, reduction = 'mean')

    return loss, acc


# def weighted_bce_nll_loss(logits, labels, pos_weight):
#     """
#     自定义加权损失函数，用于多类别输出中的二元交叉熵损失
#     Args:
#         logits: 模型的logits输出 (N, 65)，未经过Softmax或LogSoftmax
#         labels: 真实标签，形状 (N,)
#         pos_weight: 正样本的加权因子，用于平衡类别不均衡
#
#     Returns:
#         loss: 计算得到的损失值
#     """
#     # 将logits转换为概率分布
#     probs = F.softmax(logits, dim=-1)  # (N, 65)
#     N, C = logits.shape  # N: 样本数量, C: 类别数量(65)
#
#     # 为每个类别创建 one-hot 标签
#     one_hot_labels = F.one_hot(labels, num_classes=C).float()  # (N, 65)
#
#     # 计算每个类别的权重
#     weights = pos_weight * one_hot_labels + 1  # 对正样本加权
#
#     # 计算加权二元交叉熵损失
#     bce_loss = - (weights * (one_hot_labels * torch.log(probs + 1e-8) +
#                              (1 - one_hot_labels) * torch.log(1 - probs + 1e-8))).sum(dim=-1)
#
#     # 返回平均损失
#     return bce_loss.mean()
#
# def alike_distill_loss(kpts, img):
#     """
#     替换后的基于Weighted BCE的蒸馏损失
#     Args:
#         kpts: 模型输出的logits, shape (C, H, W)
#         img: 原始图像 (H, W)
#
#     Returns:
#         loss: 自定义加权二元交叉熵损失
#         acc: 预测准确率
#     """
#     C, H, W = kpts.shape
#     kpts = kpts.permute(1, 2, 0)  # 调整为 (H, W, C)
#     img = img.permute(1, 2, 0).expand(-1, -1, 3).cpu().numpy() * 255
#
#     with torch.no_grad():
#         alike_kpts = torch.tensor(extract_alike_kpts(img), device=kpts.device)
#         labels = torch.ones((H, W), dtype=torch.long, device=kpts.device) * 64  # 默认非关键点
#         offsets = (((alike_kpts / 8) - (alike_kpts / 8).long()) * 8).long()
#         offsets = offsets[:, 0] + 8 * offsets[:, 1]  # 线性索引
#         labels[(alike_kpts[:, 1] / 8).long(), (alike_kpts[:, 0] / 8).long()] = offsets
#
#     kpts = kpts.view(-1, C)  # 展平 keypoints
#     labels = labels.view(-1)  # 展平标签
#
#     # 筛选正负样本
#     mask = labels < 64
#     idxs_pos = mask.nonzero().flatten()
#     idxs_neg = (~mask).nonzero().flatten()
#     perm = torch.randperm(idxs_neg.size(0))[:len(idxs_pos) // 32]
#     idxs_neg = idxs_neg[perm]
#     idxs = torch.cat([idxs_pos, idxs_neg])
#
#     kpts = kpts[idxs]
#     labels = labels[idxs]
#
#     # 动态计算 pos_weight
#     num_pos = len(idxs_pos)
#     num_neg = len(idxs_neg)
#     pos_weight = num_neg / num_pos if num_pos > 0 else 1.0
#
#     # 计算自定义加权BCE损失
#     loss = weighted_bce_nll_loss(kpts, labels, pos_weight=pos_weight)
#
#     # 计算准确率
#     with torch.no_grad():
#         predicted = kpts.max(dim=-1)[1]
#         acc = (labels == predicted).sum() / len(labels)
#
#     return loss, acc


def keypoint_position_loss(kpts1, kpts2, pts1, pts2, softmax_temp = 1.0):
    '''
        Computes coordinate classification loss, by re-interpreting the 64 bins to 8x8 grid and optimizing
        for correct offsets
    '''
    C, H, W = kpts1.shape
    kpts1 = kpts1.permute(1,2,0) * softmax_temp
    kpts2 = kpts2.permute(1,2,0) * softmax_temp

    with torch.no_grad():
        #Generate meshgrid
        x, y = torch.meshgrid(torch.arange(W, device=kpts1.device), torch.arange(H, device=kpts1.device), indexing ='xy')
        xy = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1)], dim=-1)
        xy*=8

        #Generate collision map
        hashmap = torch.ones((H*8, W*8, 2), dtype = torch.long, device = kpts1.device) * -1
        hashmap[(pts1[:,1]).long(), (pts1[:,0]).long(), :] = (pts2).long()

        #Estimate offset of src kpts 
        _, kpts1_offsets = kpts1.max(dim=-1)
        kpts1_offsets_x = kpts1_offsets  % 8
        kpts1_offsets_y = kpts1_offsets // 8
        kpts1_offsets_xy = torch.cat([kpts1_offsets_x.unsqueeze(-1), 
                                      kpts1_offsets_y.unsqueeze(-1)], dim=-1)
        #pdb.set_trace()
        kpts1_coords = xy + kpts1_offsets_xy

        #find src -> tgt pts
        kpts1_coords = kpts1_coords.view(-1,2)
        gt_12 = hashmap[kpts1_coords[:,1], kpts1_coords[:,0]]
        mask_valid = torch.all(gt_12 >= 0, dim=-1)
        gt_12 = gt_12[mask_valid]

        #find offset labels
        labels2 = (gt_12/8) - (gt_12/8).long()
        labels2 = (labels2 * 8).long()
        labels2 = labels2[:, 0] + 8*labels2[:, 1] #linear index
        
    kpts2_selected = kpts2[(gt_12[:, 1]/8).long(), (gt_12[:, 0]/8).long()]        

    kpts1_selected = F.log_softmax(kpts1.view(-1,C)[mask_valid], dim=-1)
    kpts2_selected = F.log_softmax(kpts2_selected, dim=-1)

    #Here we enforce softmax to keep current max on src kps
    with torch.no_grad():
        _, labels1 =  kpts1_selected.max(dim=-1)

    predicted2 = kpts2_selected.max(dim=-1)[1]
    acc =  (labels2 == predicted2)
    acc = acc.sum() / len(acc)

    loss = F.nll_loss(kpts1_selected, labels1, reduction = 'mean') + \
           F.nll_loss(kpts2_selected, labels2, reduction = 'mean')
    
    #pdb.set_trace()

    return loss, acc

# def coordinate_classification_loss(coords1, pts1, pts2, conf):
#     '''
#         Computes the fine coordinate classification loss, by re-interpreting the 64 bins to 8x8 grid and optimizing
#         for correct offsets after warp
#     '''
#     #Do not backprop coordinate warps
#     with torch.no_grad():
#
#         coords1_detached = pts1 * 8
#
#         #find offset
#         offsets1_detached = (coords1_detached/8) - (coords1_detached/8).long()
#         offsets1_detached = (offsets1_detached * 8).long()
#         labels1 = offsets1_detached[:, 0] + 8*offsets1_detached[:, 1]
#
#     #pdb.set_trace()
#     coords1_log = F.log_softmax(coords1, dim=-1)
#
#     predicted = coords1.max(dim=-1)[1]
#     acc =  (labels1 == predicted)
#     acc = acc[conf > 0.1]
#     acc = acc.sum() / len(acc)
#
#     loss = F.nll_loss(coords1_log, labels1, reduction = 'none')
#
#     #Weight loss by confidence, giving more emphasis on reliable matches
#     conf = conf / conf.sum()
#     loss = (loss * conf).sum()
#
#     return loss * 2., acc

def coordinate_classification_loss(coords1, pts1, pts2, conf, num_bins=8):
    '''
    Computes the coordinate classification loss, redefining as horizontal
    and vertical classification tasks.
    '''
    # 获取batch大小
    with torch.no_grad():

        coords1_detached = pts1 * 8

        #find offset
        offsets1_detached = (coords1_detached/8) - (coords1_detached/8).long()
        offsets1_detached = (offsets1_detached * 8).long()
        # labels1 = offsets1_detached[:, 0] + 8*offsets1_detached[:, 1]
        offsets1_detached_x = offsets1_detached[:, 0]
        offsets1_detached_y = offsets1_detached[:, 1]

    # 计算标签，确保在0到num_bins-1范围内
    labels_x = torch.clamp(offsets1_detached_x, 0, num_bins - 1)
    labels_y = torch.clamp(offsets1_detached_y, 0, num_bins - 1)

    # 假设 coords1 的形状是 [batch_size, 2 * num_bins]
    logits_x = coords1[:, :num_bins]  # 提取水平坐标部分
    logits_y = coords1[:, num_bins:]  # 提取垂直坐标部分

    # 对水平方向和垂直方向分别进行 softmax
    coords1_log_x = F.log_softmax(logits_x, dim=-1)
    coords1_log_y = F.log_softmax(logits_y, dim=-1)

    # 分别提取水平和垂直方向的预测
    predicted_x = coords1_log_x.max(dim=-1)[1]
    predicted_y = coords1_log_y.max(dim=-1)[1]
    # 计算softmax和损失
    # coords1_log = F.log_softmax(coords1, dim=-1)

    # 分别提取水平和垂直方向的预测
    # predicted_x = coords1_log[:, :num_bins].max(dim=-1)[1]  # 水平方向的预测
    # predicted_y = coords1_log[:, num_bins:].max(dim=-1)[1]  # 垂直方向的预测

    # 计算准确率
    acc_x = (labels_x == predicted_x).float().mean()
    acc_y = (labels_y == predicted_y).float().mean()

    # 计算NLL损失
    loss_x = F.nll_loss(logits_x, labels_x, reduction='mean')
    loss_y = F.nll_loss(logits_y, labels_y, reduction='mean')

    conf = conf / conf.sum()  # 归一化置信度
    loss_x = (loss_x * conf).sum()  # 加权损失
    loss_y = (loss_y * conf).sum()  # 加权损失

    # 综合损失
    loss = (loss_x + loss_y) / 2
    return loss * 2., (acc_x + acc_y) / 2  # 返回总损失和平均准确率


def corner_loss(pred_offsets, gt_offsets):
    return torch.mean((pred_offsets - gt_offsets) ** 2)


import torch


def reprojection_loss(H_pred, src_points1, tgt_points1, conf):
    """
    计算重投影损失，基于单应性矩阵和源点与目标点之间的差异。

    :param H_pred: 预测的单应性矩阵，形状为 [batch_size, 3, 3]
    :param src_points: 源点，形状为 [batch_size, 8]，表示每个批次的四个角点的 x, y 坐标
    :param tgt_points: 目标点，形状为 [batch_size, 8]，表示每个批次的四个角点的 x, y 坐标
    :param conf: 置信度，形状为 [batch_size]，用于加权损失
    :return: 重投影损失
    """
    with torch.no_grad():
        coords2_detached = tgt_points1 * 8
        # find offset
        offsets2_detached = (coords2_detached / 8) - (coords2_detached / 8).long()
        tgt_points = (offsets2_detached * 8).long()

    m = H_pred.shape[0]
    src_points_local = torch.tensor([[1, 1]], dtype=torch.float32, device=H_pred.device).expand(m, 1, 2)
    src_points_hom = torch.cat([src_points_local, torch.ones(m, 1, 1, device=H_pred.device)], dim=-1)
    pred_points_hom = torch.bmm(H_pred, src_points_hom.transpose(1, 2))
    z = torch.clamp(pred_points_hom[:, 2:3, :], min=1e-6, max=1e6)
    pred_points = pred_points_hom[:, :2, :] / z
    pred_points = pred_points.squeeze(-1)

    tgt_points_local = tgt_points

    conf = conf / conf.sum()
    conf = conf.view(m, 1)  # [batch_size, 1]
    # 计算欧氏距离（L2 范数）
    reproj_error = torch.norm(pred_points - tgt_points_local, dim=1)  # [batch_size,]
    # print("reproj_error", reproj_error)
    loss = (reproj_error * conf.squeeze(1)).sum()  # 加权求和
    # conf = conf.view(m, 1).expand(m, 2)
    # reproj_error = torch.abs(pred_points - tgt_points_local)
    # loss=(reproj_error * conf).sum()
    # # loss = torch.mean(reproj_error * conf)

    return loss


def keypoint_loss(heatmap, target):
    # Compute L1 loss
    L1_loss = F.l1_loss(heatmap, target)
    return L1_loss * 3.0

def hard_triplet_loss(X,Y, margin = 0.5):

    if X.size() != Y.size() or X.dim() != 2 or Y.dim() != 2:
        raise RuntimeError('Error: X and Y shapes must match and be 2D matrices')

    dist_mat = torch.cdist(X, Y, p=2.0)
    dist_pos = torch.diag(dist_mat)
    dist_neg = dist_mat + 100.*torch.eye(*dist_mat.size(), dtype = dist_mat.dtype, 
            device = dist_mat.get_device() if dist_mat.is_cuda else torch.device("cpu"))

    #filter repeated patches on negative distances to avoid weird stuff on gradients
    dist_neg = dist_neg + dist_neg.le(0.01).float()*100.

    #Margin Ranking Loss
    hard_neg = torch.min(dist_neg, 1)[0]

    loss = torch.clamp(margin + dist_pos - hard_neg, min=0.)

    return loss.mean()
