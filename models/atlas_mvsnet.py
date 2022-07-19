import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoder import *
from .module import *

from models.features.HABNet import habnet as FeatureNet

Align_Corners_Range = False

#uses output resolution of 1/4 and optional 3D attention
#ATLAS-MVSNet
class ATLASMVSNet(nn.Module):
    def __init__(self, ndepths = [32, 8, 8, 8, 4], output_scaling=1, num_blocks=5, num_heads=1, num_channels=36):
        super(ATLASMVSNet, self).__init__()
        self.output_scaling = output_scaling
        self.ndepths = ndepths

        self.grad_method = "detach"

        c = num_channels
        # num_blocks = 5
        # num_heads = 1

        self.num_stage = len(self.ndepths)
        self.depth_intervals_ratio = [8, 4, 2, 1]

        if (self.num_stage == 4):
            self.decoders = torch.nn.ModuleList([decoderBlock(num_blocks,c,c, att=True, heads=num_heads),
                                decoderBlock(num_blocks,c,c, att=False, heads=num_heads),
                                decoderBlock(num_blocks,c,c, att=False, heads=num_heads),
                                decoderBlock(num_blocks,c,c, att=False, heads=num_heads)])
        else:
            self.decoders = torch.nn.ModuleList([decoderBlock(num_blocks,c,c, att=True, heads=num_heads),
                               decoderBlock(num_blocks,c,c, att=False, heads=num_heads),
                               decoderBlock(num_blocks,c,c, att=False, heads=num_heads),
                               decoderBlock(num_blocks,c,c, att=False, heads=num_heads),
                               decoderBlock(num_blocks,c,c, att=False, heads=num_heads)])
            self.depth_intervals_ratio = [16, 8, 4, 2, 1]

        self.stage_infos = {
            "stage0":{
                "scale": 4,
            },
            "stage1": {
                "scale": 8,
            },
            "stage2": {
                "scale": 16,
            },
            "stage3": {
                "scale": 32,
            },
            "stage4": {
                "scale": 64,
            }
        }

        self.feature_net = FeatureNet(stages=self.num_stage, num_chan=c, heads=num_heads)

        print(f"created ATLASMVSNet with: {self.num_stage} stages, {c} channels, {num_heads} heads")

    def get_volume_variance(self, features, proj_matrices, depth_values, num_depth):
        proj_matrices = torch.unbind(proj_matrices, 1)
        assert len(features) == len(proj_matrices), "Different number of images and projection matrices"
        assert depth_values.shape[1] == num_depth, "depth_values.shape[1]:{}  num_depth:{}".format(depth_values.shapep[1], num_depth)
        num_views = len(features)

        # step 1. feature extraction
        ref_feature, src_features = features[0], features[1:]
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]

        # step 2. differentiable homograph, build cost volume
        ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, num_depth, 1, 1)
        volume_sum = ref_volume
        volume_sq_sum = ref_volume ** 2
        del ref_volume
        for src_fea, src_proj in zip(src_features, src_projs):
            #warpped features
            src_proj_new = src_proj[:, 0].clone()
            src_proj_new[:, :3, :4] = torch.matmul(src_proj[:, 1, :3, :3], src_proj[:, 0, :3, :4])
            ref_proj_new = ref_proj[:, 0].clone()
            ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_proj[:, 0, :3, :4])
            warped_volume = homo_warping_ms(src_fea, src_proj_new, ref_proj_new, depth_values)

            if self.training:
                volume_sum = volume_sum + warped_volume
                volume_sq_sum = volume_sq_sum + warped_volume ** 2
            else:
                # TODO: this is only a temporal solution to save memory, better way?
                volume_sum += warped_volume
                volume_sq_sum += warped_volume.pow_(2)  # the memory of warped_volume has been modified

            del warped_volume
        # aggregate multiple feature volumes by variance
        volume_variance = volume_sq_sum.div_(num_views).sub_(volume_sum.div_(num_views).pow_(2))

        return volume_variance

    def forward(self, imgs, proj_matrices, depth_values_init):
        depth_min = float(depth_values_init[0, 0].cpu().numpy())
        depth_max = float(depth_values_init[0, -1].cpu().numpy())
        depth_interval = (depth_max - depth_min) / depth_values_init.size(1)

        # print("CUDA mem usage: ", torch.cuda.memory_allocated())

        b, n, c, output_h, output_w = imgs.shape
        output_h = output_h // self.output_scaling
        output_w = output_w // self.output_scaling

        print("output h w: ", output_h, output_w)

        #feature extraction
        features = []
        for nview_idx in range(imgs.size(1)):  #imgs shape (B, N, C, H, W)
            img = imgs[:, nview_idx]
            features.append(self.feature_net(img))

        print("CUDA mem usage for feature extraction: ", torch.cuda.memory_allocated())

        tensor_depth_min = depth_values_init[:, 0]  # (B,)
        tensor_depth_max = depth_values_init[:, -1]

        depth, cur_depth = None, None
        feat_2x = None
        entropy = None
        stacked = []

        for stage_idx in range(self.num_stage):
            # print("*********************stage{}*********************".format(stage_idx + 1))

            current_stage = self.num_stage - (stage_idx + 1) #reverse order, starting from coarsest stage N-1

            features_stage = [feat["stage{}".format(current_stage)] for feat in features]
            proj_matrices_stage = proj_matrices["stage{}".format(current_stage)]
            stage_scale = self.stage_infos["stage{}".format(current_stage)]["scale"]

            if depth is not None:
                if self.grad_method == "detach":
                    cur_depth = depth.detach()
                else:
                    cur_depth = depth
                cur_depth = F.interpolate(cur_depth.unsqueeze(1),
                                                [output_h, output_w], mode='bilinear',
                                                align_corners=Align_Corners_Range).squeeze(1)
                # print("depth: ", depth.data)
            else:
                cur_depth = depth_values_init
            depth_range_samples = get_depth_range_samples(cur_depth=cur_depth,
                                                        ndepth=self.ndepths[stage_idx],
                                                        depth_interval_pixel=self.depth_intervals_ratio[stage_idx] * depth_interval,
                                                        dtype=img.dtype,
                                                        device=img.device,
                                                        shape=[b, output_h, output_w], #img.shape[2], img.shape[3]],
                                                        max_depth=depth_max,
                                                        min_depth=depth_min)

            depth_tmp = F.interpolate(depth_range_samples.unsqueeze(1),
                            [self.ndepths[stage_idx], img.shape[2]//int(stage_scale), img.shape[3]//int(stage_scale)], mode='trilinear',
                            align_corners=Align_Corners_Range).squeeze(1)

            depth_tmp_up = F.interpolate(depth_range_samples.unsqueeze(1),
                            [self.ndepths[stage_idx], output_h, output_w], mode='trilinear',
                            align_corners=Align_Corners_Range).squeeze(1)

            volume_variance = self.get_volume_variance(features_stage, proj_matrices_stage, 
                                                depth_values=depth_tmp,
                                                num_depth=self.ndepths[stage_idx])


            cost = self.decoders[stage_idx](volume_variance)
            print("cost shape: ", cost.shape)
            cost = F.interpolate(cost, [output_h, output_w], mode='bilinear')

            if not self.training and stage_idx == (self.num_stage - 1):
                depth, entropy = depth_regression(F.softmax(cost,1), depth_values=depth_tmp_up, confidence=True)
            else:
                depth = depth_regression(F.softmax(cost,1), depth_values=depth_tmp_up)
            
            stacked.append(depth)
            
            print("CUDA mem usage after stage{}: ".format(stage_idx + 1), torch.cuda.memory_allocated())

        stacked.reverse()

        if self.training:
            entropy = stacked[0]
            return stacked, entropy
        else:
            return depth, torch.squeeze(entropy)


def atlas_loss(stacked, depth_gt, mask):
    mask = mask > 0.5
    mask.detach_()

    if len(stacked) == 5:
        # print("5 loss")
        loss = (16./31)*F.smooth_l1_loss(stacked[0][mask], depth_gt[mask], reduction='mean') + \
                (8./31)*F.smooth_l1_loss(stacked[1][mask], depth_gt[mask], reduction='mean') + \
                (4./31)*F.smooth_l1_loss(stacked[2][mask], depth_gt[mask], reduction='mean') + \
                (2./31)*F.smooth_l1_loss(stacked[3][mask], depth_gt[mask], reduction='mean') + \
                (1./31)*F.smooth_l1_loss(stacked[4][mask], depth_gt[mask], reduction='mean')
    else:
        # print("4 loss")
        loss =  (8./15)*F.smooth_l1_loss(stacked[0][mask], depth_gt[mask], reduction='mean') + \
                (4./15)*F.smooth_l1_loss(stacked[1][mask], depth_gt[mask], reduction='mean') + \
                (2./15)*F.smooth_l1_loss(stacked[2][mask], depth_gt[mask], reduction='mean') + \
                (1./15)*F.smooth_l1_loss(stacked[3][mask], depth_gt[mask], reduction='mean')

    return loss