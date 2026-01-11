import torch.nn as nn

from .common import PositionalEncoding


class StyleEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        # Model parameters
        # Fixed to 51-dim: 50 expr + 1 jaw (as per baseline_prompt.md requirements)
        if args.rot_repr == 'aa':
            if args.no_head_pose:
                self.motion_coef_dim = 51  # 50 expr + 1 jaw
            else:
                # Legacy support: 50 expr + 4 pose (head pose + jaw)
                self.motion_coef_dim = 54
        else:
            raise ValueError(f'Unknown rotation representation {args.rot_repr}!')

        self.feature_dim = args.feature_dim
        self.n_heads = args.n_heads
        self.n_layers = args.n_layers
        self.mlp_ratio = args.mlp_ratio

        # Transformer for feature extraction
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.feature_dim, nhead=self.n_heads, dim_feedforward=self.mlp_ratio * self.feature_dim,
            activation='gelu', batch_first=True
        )

        self.PE = PositionalEncoding(self.feature_dim)
        self.encoder = nn.ModuleDict({
            'motion_proj': nn.Linear(self.motion_coef_dim, self.feature_dim),
            'transformer': nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers),
        })

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, motion_coef):
        """
        :param motion_coef: (batch_size, seq_len, motion_coef_dim)
        :param audio: (batch_size, seq_len)
        :return: (batch_size, feature_dim)
        """
        batch_size, seq_len, _ = motion_coef.shape

        # Motion
        motion_feat = self.encoder['motion_proj'](motion_coef)
        motion_feat = self.PE(motion_feat)

        feat = self.encoder['transformer'](motion_feat)  # (N, L, feat_dim)

        feat = feat.mean(dim=1)  # Pooling to (N, feat_dim)

        return feat
