import math
from abc import ABC
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.functional import l1_loss
from torchvision import utils as vutils

from utils.box_utils import (box_cxcywh_to_xyxy, generalized_box_iou,
                             giou_loss, xywh2xyxy)


def sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    batch_size = inputs.shape[0]
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    prob = inputs
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.mean(1).sum() / batch_size


def dice_loss(inputs, targets):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    batch_size = inputs.shape[0]
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / batch_size


def trans_vg_loss(batch_pred, batch_target, pred_masks=None, gt_masks=None, alpha=1):
    """Compute the losses related to the bounding boxes,
    including the L1 regression loss and the GIoU loss
    """
    batch_size = batch_pred.shape[0]
    num_boxes = batch_size

    loss_bbox = F.l1_loss(batch_pred, batch_target, reduction="none")
    loss_giou = 1 - torch.diag(generalized_box_iou(xywh2xyxy(batch_pred), xywh2xyxy(batch_target)))

    losses = {}
    losses["loss_bbox"] = (loss_bbox.sum() / num_boxes) * alpha
    losses["loss_giou"] = (loss_giou.sum() / num_boxes) * alpha
    if pred_masks is not None and gt_masks is not None:
        focal_loss = sigmoid_focal_loss(pred_masks, gt_masks)
        dice_loss_ = dice_loss(pred_masks, gt_masks)
        losses["focal_loss"] = focal_loss
        losses["dice_loss"] = dice_loss_

    return losses


def edvg_s_loss(pred_dict, gt_dict, gt_masks, focal_loss, args):

    losses = {}
    focal_loss_mask = sigmoid_focal_loss(pred_dict["pred_mask"], gt_masks)
    dice_loss_mask = dice_loss(pred_dict["pred_mask"], gt_masks)
    losses["focal_loss_seg"] = focal_loss_mask
    losses["dice_loss"] = dice_loss_mask

    return losses


def edvg_l_loss(args, pred_dict, gt_dict, gt_masks, focal_loss, patch_size=16):

    losses = {}
    # gt gaussian map``
    gt_bbox = gt_dict  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
    gt_gaussian_maps = generate_heatmap(gt_dict.unsqueeze(0), 448, patch_size)  # cxcy0wh
    gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)

    # Get boxes
    pred_boxes = pred_dict["pred_boxes"]
    if torch.isnan(pred_boxes).any():
        raise ValueError("Network outputs is NAN! Stop Training")
    num_queries = pred_boxes.size(1)
    pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
    gt_boxes_vec = box_cxcywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0, max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
    # compute giou and iou
    g_loss, iou = giou_loss(pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
    # compute l1 loss
    l_loss = l1_loss(pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
    # compute location loss
    location_loss = focal_loss(pred_dict["score_map"], gt_gaussian_maps)

    losses["giou_loss"] = g_loss * args.loss_alpha
    losses["l1_loss"] = l_loss * args.loss_alpha
    losses["focal_loss_box"] = location_loss * args.loss_alpha
    return losses


def trans_vg_convhead_loss(args, pred_dict, gt_dict, gt_masks, focal_loss, patch_size=16):

    losses = {}
    if args.is_rec:
        # gt gaussian map``
        gt_bbox = gt_dict  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
        gt_gaussian_maps = generate_heatmap(gt_dict.unsqueeze(0), args.imsize, patch_size)  # cxcy0wh
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)

        # Get boxes
        pred_boxes = pred_dict["pred_boxes"]
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_cxcywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0, max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)

        # compute giou and iou
        g_loss, iou = giou_loss(pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        # compute l1 loss
        l_loss = l1_loss(pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        # compute location loss
        location_loss = focal_loss(pred_dict["score_map"], gt_gaussian_maps)

        losses["giou_loss"] = g_loss * args.loss_alpha  # 0.1
        losses["l1_loss"] = l_loss * args.loss_alpha
        losses["focal_loss_box"] = location_loss * args.loss_alpha

    if args.is_res:
        focal_loss_mask = sigmoid_focal_loss(pred_dict["pred_mask"], gt_masks)
        dice_loss_mask = dice_loss(pred_dict["pred_mask"], gt_masks)
        losses["focal_loss_seg"] = focal_loss_mask
        losses["dice_loss"] = dice_loss_mask

    return losses


def trans_vg_convhead_bl50_bi20_loss(pred_dict, gt_dict, gt_masks, focal_loss, alpha=1):
    # gt gaussian map``
    gt_bbox = gt_dict  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
    gt_gaussian_maps = generate_heatmap(gt_dict.unsqueeze(0), 448, 16)  # cxcy0wh
    gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)

    # Get boxes
    pred_boxes = pred_dict["pred_boxes"]
    if torch.isnan(pred_boxes).any():
        raise ValueError("Network outputs is NAN! Stop Training")
    num_queries = pred_boxes.size(1)
    pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
    gt_boxes_vec = box_cxcywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0, max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)

    # compute giou and iou
    g_loss, iou = giou_loss(pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
    # compute l1 loss
    l_loss = l1_loss(pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
    # compute location loss
    location_loss = focal_loss(pred_dict["score_map"], gt_gaussian_maps)

    losses = {}
    if "pred_mask" in pred_dict:
        focal_loss_mask = sigmoid_focal_loss(pred_dict["pred_mask"], gt_masks)
        dice_loss_mask = dice_loss(pred_dict["pred_mask"], gt_masks)
        losses["focal_loss_seg"] = focal_loss_mask
        losses["dice_loss"] = dice_loss_mask
    losses["giou_loss"] = g_loss * 2.0 * alpha
    losses["l1_loss"] = l_loss * 5.0 * alpha
    losses["focal_loss_box"] = location_loss * alpha
    return losses


def trans_vg_convhead_locf01_loss(pred_dict, gt_dict, gt_masks, focal_loss, alpha=1):
    # gt gaussian map``
    gt_bbox = gt_dict  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
    gt_gaussian_maps = generate_heatmap(gt_dict.unsqueeze(0), 448, 16)  # cxcy0wh
    gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)
    # np_array=(gt_gaussian_maps[-1][0].cpu().numpy()*255).astype(np.uint8)
    # np_array=(pred_dict['score_map'][0][0].cpu().detach().numpy()*255).astype(np.uint8)
    # cv2.imwrite('response.png', np_array)

    # Get boxes
    pred_boxes = pred_dict["pred_boxes"]
    if torch.isnan(pred_boxes).any():
        raise ValueError("Network outputs is NAN! Stop Training")
    num_queries = pred_boxes.size(1)
    pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
    gt_boxes_vec = box_cxcywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0, max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)

    # compute giou and iou
    g_loss, iou = giou_loss(pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
    # compute l1 loss
    l_loss = l1_loss(pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
    # compute location loss
    location_loss = focal_loss(pred_dict["score_map"], gt_gaussian_maps)

    losses = {}
    if "pred_mask" in pred_dict:
        focal_loss_mask = sigmoid_focal_loss(pred_dict["pred_mask"], gt_masks)
        dice_loss_mask = dice_loss(pred_dict["pred_mask"], gt_masks)
        losses["focal_loss_seg"] = focal_loss_mask
        losses["dice_loss"] = dice_loss_mask
    losses["giou_loss"] = g_loss * alpha
    losses["l1_loss"] = l_loss * alpha
    losses["focal_loss_box"] = location_loss * alpha * 0.1
    return losses


def trans_vg_convhead_segf05_locf05_loss(pred_dict, gt_dict, gt_masks, focal_loss, alpha=1):
    # gt gaussian map``
    gt_bbox = gt_dict  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
    gt_gaussian_maps = generate_heatmap(gt_dict.unsqueeze(0), 448, 16)  # cxcy0wh
    gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)
    # np_array=(gt_gaussian_maps[-1][0].cpu().numpy()*255).astype(np.uint8)
    # np_array=(pred_dict['score_map'][0][0].cpu().detach().numpy()*255).astype(np.uint8)
    # cv2.imwrite('response.png', np_array)

    # Get boxes
    pred_boxes = pred_dict["pred_boxes"]
    if torch.isnan(pred_boxes).any():
        raise ValueError("Network outputs is NAN! Stop Training")
    num_queries = pred_boxes.size(1)
    pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
    gt_boxes_vec = box_cxcywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0, max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)

    # compute giou and iou
    g_loss, iou = giou_loss(pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
    # compute l1 loss
    l_loss = l1_loss(pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
    # compute location loss
    location_loss = focal_loss(pred_dict["score_map"], gt_gaussian_maps)

    losses = {}
    if "pred_mask" in pred_dict:
        focal_loss_mask = sigmoid_focal_loss(pred_dict["pred_mask"], gt_masks)
        dice_loss_mask = dice_loss(pred_dict["pred_mask"], gt_masks)
        losses["focal_loss_seg"] = focal_loss_mask * 0.5
        losses["dice_loss"] = dice_loss_mask
    losses["giou_loss"] = g_loss * alpha
    losses["l1_loss"] = l_loss * alpha
    losses["focal_loss_box"] = location_loss * alpha * 0.5
    return losses


def trans_vg_convhead_segf05_loss(pred_dict, gt_dict, gt_masks, focal_loss, alpha=1):
    # gt gaussian map``
    gt_bbox = gt_dict  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
    gt_gaussian_maps = generate_heatmap(gt_dict.unsqueeze(0), 448, 16)  # cxcy0wh
    gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)
    # np_array=(gt_gaussian_maps[-1][0].cpu().numpy()*255).astype(np.uint8)
    # np_array=(pred_dict['score_map'][0][0].cpu().detach().numpy()*255).astype(np.uint8)
    # cv2.imwrite('response.png', np_array)

    # Get boxes
    pred_boxes = pred_dict["pred_boxes"]
    if torch.isnan(pred_boxes).any():
        raise ValueError("Network outputs is NAN! Stop Training")
    num_queries = pred_boxes.size(1)
    pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
    gt_boxes_vec = box_cxcywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0, max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)

    # compute giou and iou
    g_loss, iou = giou_loss(pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
    # compute l1 loss
    l_loss = l1_loss(pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
    # compute location loss
    location_loss = focal_loss(pred_dict["score_map"], gt_gaussian_maps)

    losses = {}
    if "pred_mask" in pred_dict:
        focal_loss_mask = sigmoid_focal_loss(pred_dict["pred_mask"], gt_masks)
        dice_loss_mask = dice_loss(pred_dict["pred_mask"], gt_masks)
        losses["focal_loss_seg"] = focal_loss_mask * 0.5
        losses["dice_loss"] = dice_loss_mask
    losses["giou_loss"] = g_loss * alpha
    losses["l1_loss"] = l_loss * alpha
    losses["focal_loss_box"] = location_loss * alpha
    return losses


def trans_vg_convhead_dice20_loss(pred_dict, gt_dict, gt_masks, focal_loss, alpha=1):
    # gt gaussian map``
    gt_bbox = gt_dict  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
    gt_gaussian_maps = generate_heatmap(gt_dict.unsqueeze(0), 448, 16)  # cxcy0wh
    gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)
    # np_array=(gt_gaussian_maps[-1][0].cpu().numpy()*255).astype(np.uint8)
    # np_array=(pred_dict['score_map'][0][0].cpu().detach().numpy()*255).astype(np.uint8)
    # cv2.imwrite('response.png', np_array)

    # Get boxes
    pred_boxes = pred_dict["pred_boxes"]
    if torch.isnan(pred_boxes).any():
        raise ValueError("Network outputs is NAN! Stop Training")
    num_queries = pred_boxes.size(1)
    pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
    gt_boxes_vec = box_cxcywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0, max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)

    # compute giou and iou
    g_loss, iou = giou_loss(pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
    # compute l1 loss
    l_loss = l1_loss(pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
    # compute location loss
    location_loss = focal_loss(pred_dict["score_map"], gt_gaussian_maps)

    losses = {}
    if "pred_mask" in pred_dict:
        focal_loss_mask = sigmoid_focal_loss(pred_dict["pred_mask"], gt_masks)
        dice_loss_mask = dice_loss(pred_dict["pred_mask"], gt_masks)
        losses["focal_loss_seg"] = focal_loss_mask
        losses["dice_loss"] = dice_loss_mask * 2.0
    losses["giou_loss"] = g_loss * alpha
    losses["l1_loss"] = l_loss * alpha
    losses["focal_loss_box"] = location_loss * alpha
    return losses


def trans_vg_convhead_05ctr_loss(pred_dict, gt_dict, gt_masks, focal_loss, alpha=1):
    # gt gaussian map``
    gt_bbox = gt_dict  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
    gt_gaussian_maps = generate_heatmap(gt_dict.unsqueeze(0), 448, 16)  # cxcy0wh
    gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)
    # np_array=(gt_gaussian_maps[-1][0].cpu().numpy()*255).astype(np.uint8)
    # np_array=(pred_dict['score_map'][0][0].cpu().detach().numpy()*255).astype(np.uint8)
    # cv2.imwrite('response.png', np_array)

    # Get boxes
    pred_boxes = pred_dict["pred_boxes"]
    if torch.isnan(pred_boxes).any():
        raise ValueError("Network outputs is NAN! Stop Training")
    num_queries = pred_boxes.size(1)
    pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
    gt_boxes_vec = box_cxcywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0, max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)

    # compute giou and iou
    g_loss, iou = giou_loss(pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
    # compute l1 loss
    l_loss = l1_loss(pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
    # compute location loss
    location_loss = focal_loss(pred_dict["score_map"], gt_gaussian_maps)

    losses = {}
    if "pred_mask" in pred_dict:
        focal_loss_mask = sigmoid_focal_loss(pred_dict["pred_mask"], gt_masks)
        dice_loss_mask = dice_loss(pred_dict["pred_mask"], gt_masks)
        losses["focal_loss_seg"] = focal_loss_mask
        losses["dice_loss"] = dice_loss_mask
    losses["giou_loss"] = g_loss * alpha
    losses["l1_loss"] = l_loss * alpha
    losses["focal_loss_box"] = location_loss * alpha * 0.5
    return losses


def generate_heatmap(bboxes, patch_size=320, stride=16):
    """
    Generate ground truth heatmap same as CenterNet
    Args:
        bboxes (torch.Tensor): shape of [num_search, bs, 4]

    Returns:
        gaussian_maps: list of generated heatmap

    """
    gaussian_maps = []
    heatmap_size = patch_size // stride
    for single_patch_bboxes in bboxes:
        bs = single_patch_bboxes.shape[0]
        gt_scoremap = torch.zeros(bs, heatmap_size, heatmap_size)
        classes = torch.arange(bs).to(torch.long)
        bbox = single_patch_bboxes * heatmap_size
        wh = bbox[:, 2:]
        centers_int = (bbox[:, :2]).round()
        CenterNetHeatMap.generate_score_map(gt_scoremap, classes, wh, centers_int, 0.7)
        gaussian_maps.append(gt_scoremap.to(bbox.device))
    return gaussian_maps


class CenterNetHeatMap(object):
    @staticmethod
    def generate_score_map(fmap, gt_class, gt_wh, centers_int, min_overlap):
        radius = CenterNetHeatMap.get_gaussian_radius(gt_wh, min_overlap)
        radius = torch.clamp_min(radius, 0)
        radius = radius.type(torch.int).cpu().numpy()
        for i in range(gt_class.shape[0]):
            channel_index = gt_class[i]
            CenterNetHeatMap.draw_gaussian(fmap[channel_index], centers_int[i], radius[i])

    @staticmethod
    def get_gaussian_radius(box_size, min_overlap):
        """
        copyed from CornerNet
        box_size (w, h), it could be a torch.Tensor, numpy.ndarray, list or tuple
        notice: we are using a bug-version, please refer to fix bug version in CornerNet
        """
        # box_tensor = torch.Tensor(box_size)
        box_tensor = box_size
        width, height = box_tensor[..., 0], box_tensor[..., 1]

        a1 = 1
        b1 = height + width
        c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
        sq1 = torch.sqrt(b1**2 - 4 * a1 * c1)
        r1 = (b1 + sq1) / 2

        a2 = 4
        b2 = 2 * (height + width)
        c2 = (1 - min_overlap) * width * height
        sq2 = torch.sqrt(b2**2 - 4 * a2 * c2)
        r2 = (b2 + sq2) / 2

        a3 = 4 * min_overlap
        b3 = -2 * min_overlap * (height + width)
        c3 = (min_overlap - 1) * width * height
        sq3 = torch.sqrt(b3**2 - 4 * a3 * c3)
        r3 = (b3 + sq3) / 2

        return torch.min(r1, torch.min(r2, r3))

    @staticmethod
    def gaussian2D(radius, sigma=1):
        # m, n = [(s - 1.) / 2. for s in shape]
        m, n = radius
        y, x = np.ogrid[-m : m + 1, -n : n + 1]

        gauss = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        gauss[gauss < np.finfo(gauss.dtype).eps * gauss.max()] = 0
        return gauss

    @staticmethod
    def draw_gaussian(fmap, center, radius, k=1):
        diameter = 2 * radius + 1
        gaussian = CenterNetHeatMap.gaussian2D((radius, radius), sigma=diameter / 6)
        gaussian = torch.Tensor(gaussian)
        x, y = int(center[0]), int(center[1])
        height, width = fmap.shape[:2]

        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)

        masked_fmap = fmap[y - top : y + bottom, x - left : x + right]
        masked_gaussian = gaussian[radius - top : radius + bottom, radius - left : radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_fmap.shape) > 0:
            masked_fmap = torch.max(masked_fmap, masked_gaussian * k)
            fmap[y - top : y + bottom, x - left : x + right] = masked_fmap
        # return fmap


class FocalLoss(nn.Module, ABC):
    def __init__(self, alpha=2, beta=4, epsilon=1e-6):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, prediction, target):
        positive_index = target.eq(1).float()
        negative_index = target.lt(1).float()

        negative_weights = torch.pow(1 - target, self.beta)
        # clamp min value is set to 1e-12 to maintain the numerical stability
        prediction = torch.clamp(prediction, 1e-12)

        positive_loss = torch.log(prediction) * torch.pow(1 - prediction, self.alpha) * positive_index
        # if torch.isnan(positive_loss).any() or torch.isinf(positive_loss).any():
        #     print("positive_loss is {}, stopping training, FocalLoss".format(positive_loss))
        epsilon = 1e-5
        negative_loss = torch.log(1 - prediction + epsilon) * torch.pow(prediction, self.alpha) * negative_weights * negative_index

        num_positive = positive_index.float().sum()
        positive_loss = positive_loss.sum()
        negative_loss = negative_loss.sum()

        if num_positive == 0:
            loss = -negative_loss
        elif torch.isnan(negative_loss).any() or torch.isinf(negative_loss).any():
            loss = -positive_loss / num_positive
            print(f"negative_loss is {negative_loss}")
        else:
            loss = -(positive_loss + negative_loss) / num_positive

        if not math.isfinite(loss):
            # vutils.save_image(positive_loss, '000_positive_loss.png')
            # vutils.save_image(negative_loss, '000_negative_loss.png')
            vutils.save_image(prediction, "000_output.png")
            vutils.save_image(target, "000_gt.png")
            print(f"-------------------------")
            print("Loss is {}, FocalLoss".format(loss))
            print(f"num_positive is {num_positive}")
            print(f"positive_loss is {positive_loss}")
            print(f"negative_loss is {negative_loss}")
            print(f"-------------------------")
            # if torch.isinf(gt_gaussian_maps).any():
            #     print(f'标签中有inf')
        return loss
