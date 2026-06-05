"""
Standalone TransUNet inference time measurement.
Does NOT import from TransUNet_DiceHD.py (which has a __future__ import issue).
Run: python inference_time.py
"""

import time
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
class ConfigDict(dict):
    """Minimal ConfigDict replacement — no ml_collections needed."""
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)
    def __setattr__(self, k, v):
        self[k] = v


# ─────────────────────────────────────────────
# ResNetV2 backbone
# ─────────────────────────────────────────────

class StdConv2d(nn.Conv2d):
    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-10)
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride, padding=1, bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride, padding=0, bias=bias)


class PreActBottleneck(nn.Module):
    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout // 4

        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = conv1x1(cin, cmid, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = conv3x3(cmid, cmid, stride, bias=False)
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or cin != cout:
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)

        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))
        return self.relu(residual + y)


class ResNetV2(nn.Module):
    def __init__(self, block_units, width_factor):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width

        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(3, width, kernel_size=7, stride=2, bias=False, padding=3)),
            ('gn', nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
        ]))

        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width, cout=width * 4, cmid=width))] +
                [(f'unit{i+2}', PreActBottleneck(cin=width * 4, cout=width * 4, cmid=width))
                 for i in range(block_units[0] - 1)]
            ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width * 4, cout=width * 8, cmid=width * 2, stride=2))] +
                [(f'unit{i+2}', PreActBottleneck(cin=width * 8, cout=width * 8, cmid=width * 2))
                 for i in range(block_units[1] - 1)]
            ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width * 8, cout=width * 16, cmid=width * 4, stride=2))] +
                [(f'unit{i+2}', PreActBottleneck(cin=width * 16, cout=width * 16, cmid=width * 4))
                 for i in range(block_units[2] - 1)]
            ))),
        ]))

    def forward(self, x):
        features = []
        b, c, in_size, _ = x.shape
        x = self.root(x)
        features.append(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(x)
        for i in range(len(self.body) - 1):
            x = self.body[i](x)
            right_size = int(in_size / 4 / (i + 1))
            if x.shape[-1] != right_size:
                pad = right_size - x.shape[-1]
                x = F.pad(x, [0, pad, 0, pad])
            features.append(x)
        x = self.body[-1](x)
        return x, features[::-1]


# ─────────────────────────────────────────────
# Transformer
# ─────────────────────────────────────────────

class Attention(nn.Module):
    def __init__(self, config, vis=False):
        super().__init__()
        self.vis = vis
        self.num_heads = config.transformer.num_heads
        self.head_size = config.hidden_size // self.num_heads
        self.all_head_size = self.num_heads * self.head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.out = nn.Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = nn.Dropout(config.transformer.attention_dropout_rate)
        self.proj_dropout = nn.Dropout(config.transformer.attention_dropout_rate)
        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_shape = x.size()[:-1] + (self.num_heads, self.head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        q = self.transpose_for_scores(self.query(hidden_states))
        k = self.transpose_for_scores(self.key(hidden_states))
        v = self.transpose_for_scores(self.value(hidden_states))

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_size)
        probs = self.softmax(scores)
        weights = probs if self.vis else None
        probs = self.attn_dropout(probs)

        ctx = torch.matmul(probs, v)
        ctx = ctx.permute(0, 2, 1, 3).contiguous()
        ctx = ctx.view(ctx.size()[:-2] + (self.all_head_size,))
        return self.proj_dropout(self.out(ctx)), weights


class Mlp(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.transformer.mlp_dim)
        self.fc2 = nn.Linear(config.transformer.mlp_dim, config.hidden_size)
        self.act = nn.functional.gelu
        self.dropout = nn.Dropout(config.transformer.dropout_rate)

    def forward(self, x):
        return self.dropout(self.fc2(self.dropout(self.act(self.fc1(x)))))


class Embeddings(nn.Module):
    def __init__(self, config, img_size):
        super().__init__()
        self.hybrid = True
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size

        self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor)
        in_channels = self.hybrid_model.width * 16

        grid_size = config.patches.grid
        patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
        n_patches = grid_size[0] * grid_size[1]   # = 32*32 = 1024

        self.patch_embeddings = nn.Conv2d(in_channels, config.hidden_size,
                                          kernel_size=patch_size, stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))
        self.dropout = nn.Dropout(config.transformer.dropout_rate)

    def forward(self, x):
        x, features = self.hybrid_model(x)
        x = self.patch_embeddings(x)
        x = x.flatten(2).transpose(-1, -2)
        x = self.dropout(x + self.position_embeddings)
        return x, features


class Block(nn.Module):
    def __init__(self, config, vis=False):
        super().__init__()
        self.attn_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x, w = self.attn(self.attn_norm(x))
        x = x + h
        h = x
        x = self.ffn(self.ffn_norm(x))
        return x + h, w


class Encoder(nn.Module):
    def __init__(self, config, vis=False):
        super().__init__()
        self.layer = nn.ModuleList([copy.deepcopy(Block(config, vis))
                                    for _ in range(config.transformer.num_layers)])
        self.encoder_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)

    def forward(self, hidden_states):
        for layer in self.layer:
            hidden_states, _ = layer(hidden_states)
        return self.encoder_norm(hidden_states)


class Transformer(nn.Module):
    def __init__(self, config, img_size):
        super().__init__()
        self.embeddings = Embeddings(config, img_size)
        self.encoder = Encoder(config)

    def forward(self, x):
        x, features = self.embeddings(x)
        x = self.encoder(x)
        return x, features


# ─────────────────────────────────────────────
# Decoder
# ─────────────────────────────────────────────

class Conv2dReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, use_batchnorm=True):
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=not use_batchnorm)
        relu = nn.ReLU(inplace=True)
        bn = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        super().__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=0, use_batchnorm=True):
        super().__init__()
        self.conv1 = Conv2dReLU(in_channels + skip_channels, out_channels, 3, padding=1, use_batchnorm=use_batchnorm)
        self.conv2 = Conv2dReLU(out_channels, out_channels, 3, padding=1, use_batchnorm=use_batchnorm)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv2(self.conv1(x))


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling_layer = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling_layer)


class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(config.hidden_size, head_channels, 3, padding=1, use_batchnorm=True)
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels
        skip_channels = list(config.skip_channels)
        for i in range(4 - len(skip_channels)):
            skip_channels.insert(0, 0)
        blocks = [DecoderBlock(ic, oc, sk)
                  for ic, oc, sk in zip(in_channels, out_channels, skip_channels)]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()
        h = w = int(math.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1).contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)
        for i, decoder_block in enumerate(self.blocks):
            skip = features[i] if (features is not None and i < len(features)) else None
            x = decoder_block(x, skip=skip)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=512, num_classes=2, zero_head=False):
        super().__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = 'seg'
        self.transformer = Transformer(config, img_size)
        self.decoder = DecoderCup(config)
        self.segmentation_head = SegmentationHead(
            in_channels=config.decoder_channels[-1],
            out_channels=config.n_classes,
            kernel_size=3,
        )
        self.config = config

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x, features = self.transformer(x)
        x = self.decoder(x, features)
        return self.segmentation_head(x)


# ─────────────────────────────────────────────
# Config — matches get_r50_b16_config() from TransUNet_DiceHD.py
# ─────────────────────────────────────────────

def get_r50_b16_config():
    config = ConfigDict()
    config.hidden_size = 768
    config.transformer = ConfigDict()
    config.transformer.mlp_dim = 2688
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1

    config.patches = ConfigDict()
    config.patches.size = (16, 16)
    config.patches.grid = (32, 32)   # 512 / 16 = 32

    config.resnet = ConfigDict()
    config.resnet.num_layers = (3, 4, 9)
    config.resnet.width_factor = 1

    config.decoder_channels = (256, 128, 64, 16)
    config.skip_channels = [512, 256, 64, 0]   # 4th block has no skip connection
    config.n_classes = 2
    config.activation = 'softmax'
    return config


def create_model():
    config = get_r50_b16_config()
    model = VisionTransformer(config, img_size=512, num_classes=2)
    model = nn.DataParallel(model)
    return model


# ─────────────────────────────────────────────
# Load weights
# ─────────────────────────────────────────────

def load_weights(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    # If saved without DataParallel prefix, add it
    first_key = next(iter(ckpt))
    if not first_key.startswith("module."):
        ckpt = OrderedDict((f"module.{k}", v) for k, v in ckpt.items())
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")
    return model


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    from pathlib import Path
    CKPT = str(Path(__file__).resolve().parent / "weights" / "DICE_HD_best_model_100after.pth")

    print("Loading model...")
    model = create_model()
    model = load_weights(model, CKPT)
    model.eval()
    model = model.cuda()
    print("Model loaded.")

    dummy = torch.randn(1, 3, 512, 512).cuda()

    # Warm up
    print("Warming up GPU (5 runs)...")
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy)

    # Time 50 runs
    print("Timing 50 inference runs...")
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(50):
            _ = model(dummy)
    torch.cuda.synchronize()
    elapsed_ms = (time.time() - start) / 50 * 1000

    total_params = sum(p.numel() for p in model.parameters()) / 1e6

    print()
    print("=" * 50)
    print(f"TransUNet inference per slice : {elapsed_ms:.1f} ms")
    print(f"Total parameters              : {total_params:.1f} million")
    print("=" * 50)
    print()
    print("Copy this into reviewer_04.md / paper Section 4:")
    print(f"  TransUNet stage: {elapsed_ms:.1f} ms per 512x512 slice")
    print(f"  Parameters: {total_params:.1f} million")
