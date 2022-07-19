import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

# from: https://github.com/leaderj1001/Stand-Alone-Self-Attention
class AttentionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super(AttentionLayer, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

        assert self.out_channels % self.groups == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

        self.rel_h = nn.Parameter(torch.randn(out_channels // 2, 1, 1, kernel_size, 1), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn(out_channels // 2, 1, 1, 1, kernel_size), requires_grad=True)

        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

        self.reset_parameters()

    def forward(self, x):
        batch, channels, height, width = x.size()

        # print("Att3d x shape: ", x.shape)

        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])
        q_out = self.query_conv(x)
        k_out = self.key_conv(padded_x)
        v_out = self.value_conv(padded_x)

        # print("Att3d k_out shape: ", k_out.shape)
        # print("Att3d rel_h shape: ", self.rel_h.shape)

        k_out = k_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        v_out = v_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)

        # print("Att2d k_out shape: ", k_out.shape)

        k_out_h, k_out_w = k_out.split(self.out_channels // 2, dim=1)
        # print("Att2d k_out_h shape: ", k_out_h.shape)
        # print("Att2d rel_h shape: ", self.rel_h.shape)
        # print("Att2d k_out_w shape: ", k_out_w.shape)
        # print("Att2d rel_w shape: ", self.rel_w.shape)

        k_out = torch.cat((k_out_h + self.rel_h, k_out_w + self.rel_w), dim=1)
        # print("Att3d k_out shape: ", k_out.shape)

        k_out = k_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)
        v_out = v_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)

        # print("Att3d k_out shape: ", k_out.shape)

        q_out = q_out.view(batch, self.groups, self.out_channels // self.groups, height, width, 1)

        # print("Att3d q_out shape: ", q_out.shape)

        out = q_out * k_out
        # print("Att3d out shape: ", out.shape)
        out = F.softmax(out, dim=-1)
        # print("Att3d softmax shape: ", out.shape)

        out = torch.einsum('b n c h w k, b n c h w k -> b n c h w', out, v_out).view(batch, -1, height, width)

        # print("Att3d sum shape: ", out.shape)

        return out

    def reset_parameters(self):
        init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.value_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')

        init.normal_(self.rel_h, 0, 1)
        init.normal_(self.rel_w, 0, 1)

#Extention for 3D
class AttentionLayer3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super(AttentionLayer3D, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

        assert self.out_channels % self.groups == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"
        assert self.out_channels % 3 == 0, f"out_channels should be divisible by 3. out_channels: {self.out_channels}"

        self.rel_h = nn.Parameter(torch.randn(out_channels // 3, 1, 1, 1, kernel_size, 1, 1), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn(out_channels // 3, 1, 1, 1, 1, kernel_size, 1), requires_grad=True)

        self.rel_d = nn.Parameter(torch.randn(out_channels // 3, 1, 1, 1, 1, 1, kernel_size), requires_grad=True)

        self.key_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.query_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.value_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=bias)

        self.reset_parameters()

    def forward(self, x):
        # print("Att3d x shape: ", x.shape)
        batch, channels, depth, height, width = x.size()

        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding, self.padding, self.padding])
        q_out = self.query_conv(x)
        k_out = self.key_conv(padded_x)
        v_out = self.value_conv(padded_x)

        # print("Att3d k_out shape: ", k_out.shape)
        # print("Att3d rel_h shape: ", self.rel_h.shape)

        k_out = k_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride).unfold(4, self.kernel_size, self.stride)
        v_out = v_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride).unfold(4, self.kernel_size, self.stride)

        # print("Att3d k_out shape: ", k_out.shape)

        k_out_d, k_out_h, k_out_w = k_out.split(self.out_channels // 3, dim=1)

        # print("Att3d k_out_d shape: ", k_out_d.shape)
        # print("Att3d rel_d shape: ", self.rel_d.shape)
        # print("Att3d k_out_h shape: ", k_out_h.shape)
        # print("Att3d rel_h shape: ", self.rel_h.shape)
        # print("Att3d k_out_w shape: ", k_out_w.shape)
        # print("Att3d rel_w shape: ", self.rel_w.shape)

        k_out = torch.cat((k_out_d + self.rel_d, k_out_h + self.rel_h, k_out_w + self.rel_w), dim=1)

        k_out = k_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, depth, height, width, -1)
        v_out = v_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, depth, height, width, -1)

        q_out = q_out.view(batch, self.groups, self.out_channels // self.groups, depth, height, width, 1)

        out = q_out * k_out
        out = F.softmax(out, dim=-1)
        out = torch.einsum('b n c d h w k, b n c d h w k -> b n c d h w', out, v_out).view(batch, -1, depth, height, width)

        return out

    def reset_parameters(self):
        init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.value_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')

        init.normal_(self.rel_h, 0, 1)
        init.normal_(self.rel_w, 0, 1)
        init.normal_(self.rel_d, 0, 1)