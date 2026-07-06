import torch
import torch.nn as nn


class PointMLPHead(nn.Module):
    def __init__(self, feature_dim=128, hidden_dim=256, num_layers=3, delta_scale=0.25,
                 use_base_point=True, rgb_feature_dim=0):
        super().__init__()
        self.delta_scale = delta_scale
        self.use_base_point = use_base_point
        self.rgb_feature_dim = rgb_feature_dim
        input_dim = 2 + 1 + 1 + feature_dim + feature_dim
        if use_base_point:
            input_dim += 2 + 2
        input_dim += rgb_feature_dim
        layers = []
        dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            dim = hidden_dim
        self.mlp = nn.Sequential(*layers)
        self.xy_head = nn.Linear(dim, 2)
        self.conf_head = nn.Linear(dim, 1)

    def forward(self, xy_norm, t_src, t_tgt, feat_src, feat_tgt, base_xy_norm=None, rgb_feat=None):
        inputs = [xy_norm, t_src, t_tgt, feat_src, feat_tgt]
        if self.use_base_point:
            if base_xy_norm is None:
                raise ValueError('base_xy_norm is required when use_base_point=True')
            inputs.extend([base_xy_norm, base_xy_norm - xy_norm])
        if self.rgb_feature_dim > 0:
            if rgb_feat is None:
                raise ValueError('rgb_feat is required when rgb_feature_dim > 0')
            inputs.append(rgb_feat)
        x = torch.cat(inputs, dim=-1)
        h = self.mlp(x)
        delta = torch.tanh(self.xy_head(h)) * self.delta_scale
        base = base_xy_norm if self.use_base_point else xy_norm
        pred_xy_norm = base + delta
        conf_logit = self.conf_head(h)
        return pred_xy_norm, conf_logit
