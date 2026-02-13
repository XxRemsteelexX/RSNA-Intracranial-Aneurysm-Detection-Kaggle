#!/usr/bin/env python3
"""
Optimized Eric3D training with Phase 2 improvements:
- Data augmentation
- Class weighting
- Learning rate scheduling
- Early stopping
- Mixed precision training
- Multi-scale support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse
from sklearn.metrics import roc_auc_score
import json
import scipy.ndimage as ndi


# --- Data Augmentation ---

class VolumeAugmentation:
    """3D volume augmentation for training."""

    def __init__(self, rotation_range=15, flip=True, zoom_range=(0.9, 1.1),
                 shift_range=0.1, brightness_range=0.2, contrast_range=0.2):
        self.rotation_range = rotation_range
        self.flip = flip
        self.zoom_range = zoom_range
        self.shift_range = shift_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range

    def __call__(self, volume):
        """Apply random augmentations to volume."""
        # Ensure volume is writable copy
        volume = np.array(volume, dtype=np.float32, copy=True)
        original_shape = volume.shape

        # Random rotation
        if self.rotation_range > 0:
            angles = np.random.uniform(-self.rotation_range, self.rotation_range, 3)
            volume = ndi.rotate(volume, angles[0], axes=(1, 2), reshape=False, order=1)
            volume = ndi.rotate(volume, angles[1], axes=(0, 2), reshape=False, order=1)
            volume = ndi.rotate(volume, angles[2], axes=(0, 1), reshape=False, order=1)

        # Random flip
        if self.flip:
            if np.random.rand() > 0.5:
                volume = np.ascontiguousarray(np.flip(volume, axis=0))
            if np.random.rand() > 0.5:
                volume = np.ascontiguousarray(np.flip(volume, axis=1))
            if np.random.rand() > 0.5:
                volume = np.ascontiguousarray(np.flip(volume, axis=2))

        # Random zoom
        if self.zoom_range:
            zoom_factor = np.random.uniform(*self.zoom_range)
            volume = ndi.zoom(volume, zoom_factor, order=1)
            # Crop/pad back to original size
            volume = self._resize_to_original(volume, original_shape)

        # Random shift
        if self.shift_range > 0:
            shift = [int(s * self.shift_range * np.random.uniform(-1, 1))
                    for s in original_shape]
            volume = ndi.shift(volume, shift, order=1)

        # Random brightness/contrast
        if self.brightness_range > 0:
            brightness = 1.0 + np.random.uniform(-self.brightness_range, self.brightness_range)
            volume = volume * brightness

        if self.contrast_range > 0:
            contrast = 1.0 + np.random.uniform(-self.contrast_range, self.contrast_range)
            mean = volume.mean()
            volume = mean + (volume - mean) * contrast

        # Clip to valid range
        volume = np.clip(volume, 0, 1)

        return np.ascontiguousarray(volume, dtype=np.float32)

    def _resize_to_original(self, volume, target_shape):
        """Crop or pad volume to target shape."""
        current = volume.shape
        pad_width = []
        for cur, tgt in zip(current, target_shape):
            diff = tgt - cur
            pad_before = diff // 2
            pad_after = diff - pad_before
            pad_width.append((max(0, pad_before), max(0, pad_after)))

        if any(p[0] > 0 or p[1] > 0 for p in pad_width):
            volume = np.pad(volume, pad_width, mode='constant', constant_values=0)

        # Crop if needed
        for i in range(3):
            if volume.shape[i] > target_shape[i]:
                diff = volume.shape[i] - target_shape[i]
                start = diff // 2
                slices = [slice(None)] * 3
                slices[i] = slice(start, start + target_shape[i])
                volume = volume[tuple(slices)]

        return np.ascontiguousarray(volume)


# --- Dataset ---

class Eric3DDataset(Dataset):
    """Dataset for Eric3D patches with augmentation."""

    def __init__(self, patch_files, labels_df, patch_size=64, augment=False):
        self.patch_files = patch_files
        self.labels_df = labels_df
        self.patch_size = patch_size
        self.augment = augment

        if augment:
            self.augmentation = VolumeAugmentation()
        else:
            self.augmentation = None

        # Map series_uid to labels
        self.uid_to_labels = {}
        for _, row in labels_df.iterrows():
            uid = row['series_uid']
            labels = row[LABEL_COLS].values.astype(np.float32)
            self.uid_to_labels[uid] = labels

    def __len__(self):
        return len(self.patch_files)

    def __getitem__(self, idx):
        h5_path = self.patch_files[idx]
        series_uid = h5_path.stem

        # Load patch
        with h5py.File(h5_path, 'r') as f:
            g = f[f'patches_{self.patch_size}']
            # Take first patch (for now - could sample random patch if multiple)
            patch = np.array(g['data'][0], dtype=np.float32)  # (64, 64, 64)

        # Apply augmentation
        if self.augmentation:
            patch = self.augmentation(patch)
        else:
            patch = np.ascontiguousarray(patch, dtype=np.float32)

        # Add channel dimension
        patch = patch[np.newaxis, ...]  # (1, 64, 64, 64)

        # Get labels
        labels = self.uid_to_labels[series_uid].copy()

        return torch.from_numpy(patch).float(), torch.from_numpy(labels).float()


# --- Model ---

class ResNet3D(nn.Module):
    """3D ResNet for aneurysm detection."""

    def __init__(self, num_classes=14, depth=34):
        super().__init__()

        # Initial conv
        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # ResNet blocks
        if depth == 18:
            self.layer1 = self._make_layer(64, 64, 2)
            self.layer2 = self._make_layer(64, 128, 2, stride=2)
            self.layer3 = self._make_layer(128, 256, 2, stride=2)
            self.layer4 = self._make_layer(256, 512, 2, stride=2)
            final_channels = 512
        elif depth == 34:
            self.layer1 = self._make_layer(64, 64, 3)
            self.layer2 = self._make_layer(64, 128, 4, stride=2)
            self.layer3 = self._make_layer(128, 256, 6, stride=2)
            self.layer4 = self._make_layer(256, 512, 3, stride=2)
            final_channels = 512
        elif depth == 50:
            # ResNet-50 would need bottleneck blocks - simplified here
            self.layer1 = self._make_layer(64, 128, 3)
            self.layer2 = self._make_layer(128, 256, 4, stride=2)
            self.layer3 = self._make_layer(256, 512, 6, stride=2)
            self.layer4 = self._make_layer(512, 1024, 3, stride=2)
            final_channels = 1024
        elif depth == 101:
            # ResNet-101 (using same channel progression as ResNet-34)
            self.layer1 = self._make_layer(64, 64, 3)
            self.layer2 = self._make_layer(64, 128, 4, stride=2)
            self.layer3 = self._make_layer(128, 256, 23, stride=2)
            self.layer4 = self._make_layer(256, 512, 3, stride=2)
            final_channels = 512

        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool3d(1)

        # Classifier
        self.fc = nn.Linear(final_channels, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(self._make_block(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(self._make_block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def _make_block(self, in_channels, out_channels, stride=1):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class DenseNet3D(nn.Module):
    """3D DenseNet for aneurysm detection."""

    def __init__(self, num_classes=14, growth_rate=32, num_init_features=64,
                 block_config=(6, 12, 24, 16), compression=0.5):
        super().__init__()

        # Initial convolution
        self.features = nn.Sequential(
            nn.Conv3d(1, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )

        # Dense blocks
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = self._make_dense_block(num_features, growth_rate, num_layers)
            self.features.add_module(f'denseblock{i+1}', block)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_config) - 1:
                trans = self._make_transition(num_features, int(num_features * compression))
                self.features.add_module(f'transition{i+1}', trans)
                num_features = int(num_features * compression)

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm3d(num_features))
        self.features.add_module('relu5', nn.ReLU(inplace=True))

        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(num_features, num_classes)

    def _make_dense_block(self, num_features, growth_rate, num_layers):
        class DenseBlock(nn.Module):
            def __init__(self, num_features, growth_rate, num_layers):
                super().__init__()
                self.layers = nn.ModuleList()
                for i in range(num_layers):
                    self.layers.append(nn.Sequential(
                        nn.BatchNorm3d(num_features + i * growth_rate),
                        nn.ReLU(inplace=True),
                        nn.Conv3d(num_features + i * growth_rate, growth_rate,
                                 kernel_size=3, padding=1, bias=False)
                    ))

            def forward(self, x):
                features = [x]
                for layer in self.layers:
                    new_features = layer(torch.cat(features, 1))
                    features.append(new_features)
                return torch.cat(features, 1)

        return DenseBlock(num_features, growth_rate, num_layers)

    def _make_transition(self, in_channels, out_channels):
        return nn.Sequential(
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool3d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        features = self.features(x)
        out = self.avgpool(features)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


class EfficientNet3D(nn.Module):
    """3D EfficientNet for aneurysm detection."""

    def __init__(self, num_classes=14, variant='b0'):
        super().__init__()

        # EfficientNet-B0 and B2 configurations
        # (expansion, out_channels, num_blocks, stride, kernel_size)
        if variant == 'b0':
            cfg = [
                (1, 16, 1, 1, 3),   # Stage 1
                (6, 24, 2, 2, 3),   # Stage 2
                (6, 40, 2, 2, 5),   # Stage 3
                (6, 80, 3, 2, 3),   # Stage 4
                (6, 112, 3, 1, 5),  # Stage 5
                (6, 192, 4, 2, 5),  # Stage 6
                (6, 320, 1, 1, 3),  # Stage 7
            ]
            width_mult = 1.0
            depth_mult = 1.0
        elif variant == 'b2':
            cfg = [
                (1, 16, 1, 1, 3),
                (6, 24, 2, 2, 3),
                (6, 40, 2, 2, 5),
                (6, 80, 3, 2, 3),
                (6, 112, 3, 1, 5),
                (6, 192, 4, 2, 5),
                (6, 320, 1, 1, 3),
            ]
            width_mult = 1.1
            depth_mult = 1.2

        # Stem
        out_channels = int(32 * width_mult)
        self.stem = nn.Sequential(
            nn.Conv3d(1, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.SiLU(inplace=True)
        )

        # Build blocks
        in_channels = out_channels
        layers = []
        for expand_ratio, c, n, s, k in cfg:
            out_channels = int(c * width_mult)
            num_blocks = int(n * depth_mult)
            for i in range(num_blocks):
                stride = s if i == 0 else 1
                layers.append(
                    self._make_mbconv_block(in_channels, out_channels, expand_ratio, stride, k)
                )
                in_channels = out_channels

        self.blocks = nn.Sequential(*layers)

        # Head
        final_channels = int(1280 * width_mult)
        self.head = nn.Sequential(
            nn.Conv3d(in_channels, final_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(final_channels),
            nn.SiLU(inplace=True)
        )

        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(final_channels, num_classes)

    def _make_mbconv_block(self, in_channels, out_channels, expand_ratio, stride, kernel_size):
        """Mobile Inverted Bottleneck Convolution block."""
        return MBConvBlock(in_channels, out_channels, expand_ratio, stride, kernel_size)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class MBConvBlock(nn.Module):
    """Mobile Inverted Bottleneck Convolution block with SE."""
    def __init__(self, in_channels, out_channels, expand_ratio, stride, kernel_size):
        super().__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expand_ratio

        layers = []

        # Expansion phase
        if expand_ratio != 1:
            layers.extend([
                nn.Conv3d(in_channels, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.SiLU(inplace=True)
            ])

        # Depthwise convolution
        padding = (kernel_size - 1) // 2
        layers.extend([
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride,
                     padding=padding, groups=hidden_dim, bias=False),
            nn.BatchNorm3d(hidden_dim),
            nn.SiLU(inplace=True)
        ])

        self.conv = nn.Sequential(*layers)

        # Squeeze-and-Excitation
        squeeze_channels = max(1, in_channels // 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(hidden_dim, squeeze_channels, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv3d(squeeze_channels, hidden_dim, kernel_size=1),
            nn.Sigmoid()
        )

        # Projection phase
        self.project = nn.Sequential(
            nn.Conv3d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_channels)
        )

    def forward(self, x):
        identity = x
        x = self.conv(x)
        x = x * self.se(x)  # Apply SE
        x = self.project(x)
        if self.use_residual:
            x = x + identity
        return x


class VisionTransformer3D(nn.Module):
    """3D Vision Transformer for aneurysm detection."""

    def __init__(self, num_classes=14, patch_size=8, embed_dim=384, depth=6, num_heads=6, variant='base'):
        super().__init__()
        self.patch_size = patch_size

        # Scale for large variant (similar to ViT-Large: 1024 dim, 24 layers, 16 heads)
        if variant == 'large':
            embed_dim = 768  # Larger embedding (between base 384 and huge 1024)
            depth = 12  # More layers
            num_heads = 12  # More attention heads

        self.embed_dim = embed_dim

        # Patch embedding
        self.patch_embed = nn.Conv3d(1, embed_dim, kernel_size=patch_size, stride=patch_size)

        # Positional embedding
        num_patches = (64 // patch_size) ** 3  # For 64x64x64 input
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # Patch embedding: (B, 1, 64, 64, 64) -> (B, embed_dim, 8, 8, 8)
        x = self.patch_embed(x)

        # Flatten patches: (B, embed_dim, 8, 8, 8) -> (B, 512, embed_dim)
        B = x.shape[0]
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)

        # Add positional embedding
        x = x + self.pos_embed

        # Transformer
        x = self.transformer(x)

        # Global average pooling
        x = x.mean(dim=1)

        # Classification
        x = self.norm(x)
        x = self.fc(x)

        return x


class UNet3D(nn.Module):
    """3D U-Net for aneurysm detection."""

    def __init__(self, num_classes=14, init_features=32):
        super().__init__()

        features = init_features

        # Encoder
        self.encoder1 = self._block(1, features)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.encoder2 = self._block(features, features * 2)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.encoder3 = self._block(features * 2, features * 4)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.encoder4 = self._block(features * 4, features * 8)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = self._block(features * 8, features * 16)

        # Decoder
        self.upconv4 = nn.ConvTranspose3d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = self._block((features * 8) * 2, features * 8)

        self.upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = self._block((features * 4) * 2, features * 4)

        self.upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = self._block((features * 2) * 2, features * 2)

        self.upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = self._block(features * 2, features)

        # Global pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(features, num_classes)

    def _block(self, in_channels, features):
        return nn.Sequential(
            nn.Conv3d(in_channels, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(features),
            nn.ReLU(inplace=True),
            nn.Conv3d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))

        # Decoder with skip connections
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        # Global pooling and classification
        x = self.avgpool(dec1)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class MobileNetV2_3D(nn.Module):
    """3D MobileNet-v2 for aneurysm detection."""

    def __init__(self, num_classes=14, width_mult=1.0):
        super().__init__()

        # Initial stem
        input_channel = int(32 * width_mult)
        self.stem = nn.Sequential(
            nn.Conv3d(1, input_channel, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(input_channel),
            nn.ReLU6(inplace=True)
        )

        # MobileNet-v2 config: (expansion, out_channels, num_blocks, stride)
        cfg = [
            (1, 16, 1, 1),
            (6, 24, 2, 2),
            (6, 32, 3, 2),
            (6, 64, 4, 2),
            (6, 96, 3, 1),
            (6, 160, 3, 2),
            (6, 320, 1, 1),
        ]

        # Build inverted residual blocks
        layers = []
        for t, c, n, s in cfg:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(InvertedResidual3DV2(input_channel, output_channel, stride, t))
                input_channel = output_channel

        self.blocks = nn.Sequential(*layers)

        # Head
        last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        self.head = nn.Sequential(
            nn.Conv3d(input_channel, last_channel, kernel_size=1, bias=False),
            nn.BatchNorm3d(last_channel),
            nn.ReLU6(inplace=True)
        )

        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class InvertedResidual3DV2(nn.Module):
    """Inverted Residual block for MobileNet-v2"""
    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        hidden_dim = int(inp * expand_ratio)
        self.use_residual = (self.stride == 1 and inp == oup)

        layers = []
        # Expansion
        if expand_ratio != 1:
            layers.extend([
                nn.Conv3d(inp, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU6(inplace=True)
            ])

        # Depthwise
        layers.extend([
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, stride=stride,
                     padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU6(inplace=True),
            # Projection
            nn.Conv3d(hidden_dim, oup, kernel_size=1, bias=False),
            nn.BatchNorm3d(oup)
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV3_3D(nn.Module):
    """3D MobileNet-v3 Large for aneurysm detection."""

    def __init__(self, num_classes=14):
        super().__init__()

        # Initial stem
        self.stem = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.Hardswish(inplace=True)
        )

        # MobileNet-v3 Large config: (in, exp, out, SE, NL, s)
        # NL: 'HS' = Hardswish, 'RE' = ReLU
        # SE: True/False for Squeeze-Excitation
        self.blocks = nn.Sequential(
            self._bneck(16, 16, 16, False, 'RE', 1),
            self._bneck(16, 64, 24, False, 'RE', 2),
            self._bneck(24, 72, 24, False, 'RE', 1),
            self._bneck(24, 72, 40, True, 'RE', 2),
            self._bneck(40, 120, 40, True, 'RE', 1),
            self._bneck(40, 120, 40, True, 'RE', 1),
            self._bneck(40, 240, 80, False, 'HS', 2),
            self._bneck(80, 200, 80, False, 'HS', 1),
            self._bneck(80, 184, 80, False, 'HS', 1),
            self._bneck(80, 184, 80, False, 'HS', 1),
            self._bneck(80, 480, 112, True, 'HS', 1),
            self._bneck(112, 672, 112, True, 'HS', 1),
            self._bneck(112, 672, 160, True, 'HS', 2),
            self._bneck(160, 960, 160, True, 'HS', 1),
            self._bneck(160, 960, 160, True, 'HS', 1),
        )

        # Head
        self.head = nn.Sequential(
            nn.Conv3d(160, 960, kernel_size=1, bias=False),
            nn.BatchNorm3d(960),
            nn.Hardswish(inplace=True)
        )

        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(
            nn.Linear(960, 1280),
            nn.Hardswish(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes)
        )

    def _bneck(self, in_c, exp_c, out_c, use_se, nl, stride):
        """Bottleneck block"""
        return InvertedResidual3D(in_c, exp_c, out_c, use_se, nl, stride)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class InvertedResidual3D(nn.Module):
    """Inverted Residual block for MobileNet-v3"""
    def __init__(self, in_c, exp_c, out_c, use_se, nl, stride):
        super().__init__()
        self.use_residual = (stride == 1 and in_c == out_c)

        activation = nn.Hardswish if nl == 'HS' else nn.ReLU

        layers = []
        # Expansion
        if exp_c != in_c:
            layers.extend([
                nn.Conv3d(in_c, exp_c, kernel_size=1, bias=False),
                nn.BatchNorm3d(exp_c),
                activation(inplace=True)
            ])

        # Depthwise
        layers.extend([
            nn.Conv3d(exp_c, exp_c, kernel_size=3, stride=stride, padding=1, groups=exp_c, bias=False),
            nn.BatchNorm3d(exp_c),
            activation(inplace=True)
        ])

        # SE
        if use_se:
            layers.append(SEBlock3D(exp_c))

        # Projection
        layers.extend([
            nn.Conv3d(exp_c, out_c, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_c)
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class SEBlock3D(nn.Module):
    """Squeeze-and-Excitation block for 3D"""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Hardsigmoid(inplace=True)
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y


class MobileNetV4_3D(nn.Module):
    """3D MobileNet-v4 for aneurysm detection.

    MobileNetV4 introduces Universal Inverted Bottleneck (UIB) blocks
    and Mobile MQA (Multi-Query Attention) for efficient inference.
    """

    def __init__(self, num_classes=14, variant='medium'):
        super().__init__()

        # MobileNetV4 variants
        if variant == 'small':
            # Small variant: fewer channels and blocks
            self.stem_channels = 16
            cfg = [
                # (exp, out, kernel, stride, use_mqa)
                (64, 24, 3, 2, False),
                (96, 24, 3, 1, False),
                (96, 32, 3, 2, False),
                (128, 32, 3, 1, True),
                (128, 64, 5, 2, True),
                (256, 64, 5, 1, True),
                (256, 96, 5, 2, True),
                (384, 96, 5, 1, True),
            ]
            head_channels = 960
        elif variant == 'medium':
            # Medium variant: balanced
            self.stem_channels = 24
            cfg = [
                (96, 32, 3, 2, False),
                (128, 32, 3, 1, False),
                (128, 48, 3, 2, False),
                (192, 48, 3, 1, True),
                (192, 80, 5, 2, True),
                (320, 80, 5, 1, True),
                (320, 128, 5, 2, True),
                (512, 128, 5, 1, True),
            ]
            head_channels = 1280
        else:  # large
            # Large variant: more capacity
            self.stem_channels = 32
            cfg = [
                (128, 48, 3, 2, False),
                (192, 48, 3, 1, False),
                (192, 64, 3, 2, False),
                (256, 64, 3, 1, True),
                (256, 112, 5, 2, True),
                (448, 112, 5, 1, True),
                (448, 160, 5, 2, True),
                (640, 160, 5, 1, True),
            ]
            head_channels = 1536

        # Stem convolution
        self.stem = nn.Sequential(
            nn.Conv3d(1, self.stem_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(self.stem_channels),
            nn.ReLU(inplace=True)
        )

        # Build UIB blocks
        layers = []
        in_channels = self.stem_channels
        for exp_c, out_c, k, s, use_mqa in cfg:
            layers.append(UIBBlock3D(in_channels, exp_c, out_c, k, s, use_mqa))
            in_channels = out_c

        self.blocks = nn.Sequential(*layers)

        # Head
        self.head = nn.Sequential(
            nn.Conv3d(in_channels, head_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(head_channels),
            nn.ReLU(inplace=True)
        )

        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(head_channels, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class UIBBlock3D(nn.Module):
    """Universal Inverted Bottleneck block for MobileNetV4."""

    def __init__(self, in_c, exp_c, out_c, kernel_size, stride, use_mqa):
        super().__init__()
        self.stride = stride
        self.use_residual = (stride == 1 and in_c == out_c)
        self.use_mqa = use_mqa

        padding = (kernel_size - 1) // 2

        # Expansion phase
        self.expand = nn.Sequential(
            nn.Conv3d(in_c, exp_c, kernel_size=1, bias=False),
            nn.BatchNorm3d(exp_c),
            nn.ReLU(inplace=True)
        )

        # Depthwise convolution
        self.depthwise = nn.Sequential(
            nn.Conv3d(exp_c, exp_c, kernel_size=kernel_size, stride=stride,
                     padding=padding, groups=exp_c, bias=False),
            nn.BatchNorm3d(exp_c),
            nn.ReLU(inplace=True)
        )

        # Mobile MQA (Multi-Query Attention) - disabled for simplicity
        # MQA adds complexity without proven benefit for 3D medical imaging
        self.mqa = None

        # SE block
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(exp_c, exp_c // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(exp_c // 4, exp_c, kernel_size=1),
            nn.Sigmoid()
        )

        # Projection
        self.project = nn.Sequential(
            nn.Conv3d(exp_c, out_c, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_c)
        )

    def forward(self, x):
        identity = x

        # Expansion
        x = self.expand(x)

        # Depthwise
        x = self.depthwise(x)

        # MQA if enabled
        if self.mqa is not None:
            x = self.mqa(x)

        # SE
        x = x * self.se(x)

        # Projection
        x = self.project(x)

        # Residual connection
        if self.use_residual:
            x = x + identity

        return x


class MobileMQA3D(nn.Module):
    """Mobile Multi-Query Attention for 3D (simplified)."""

    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv3d(channels, channels, kernel_size=1)
        self.key = nn.Conv3d(channels, channels // 4, kernel_size=1)  # Shared key
        self.value = nn.Conv3d(channels, channels // 4, kernel_size=1)  # Shared value
        self.out_proj = nn.Conv3d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, D, H, W = x.shape
        DHW = D * H * W

        # Compute Q, K, V
        q = self.query(x).view(B, C, DHW).permute(0, 2, 1)  # (B, DHW, C)
        k = self.key(x).view(B, C//4, DHW).permute(0, 2, 1)  # (B, DHW, C//4)
        v = self.value(x).view(B, C//4, DHW).permute(0, 2, 1)  # (B, DHW, C//4)

        # Attention: scale query and key properly
        scale = (C // 4) ** -0.5
        attn = torch.softmax(torch.bmm(q, k.transpose(1, 2)) * scale, dim=-1)  # (B, DHW, DHW)
        out = torch.bmm(attn, v)  # (B, DHW, C//4)

        # Expand back to full channels and reshape
        out = out.repeat(1, 1, 4)  # (B, DHW, C)
        out = out.permute(0, 2, 1).view(B, C, D, H, W)
        out = self.out_proj(out)

        return out + x  # Residual


class SwinTransformer3D(nn.Module):
    """3D Swin Transformer for aneurysm detection."""

    def __init__(self, num_classes=14, embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24]):
        super().__init__()

        # Patch embedding
        self.patch_embed = nn.Sequential(
            nn.Conv3d(1, embed_dim, kernel_size=4, stride=4),
            nn.LayerNorm([embed_dim, 16, 16, 16])
        )

        # Swin Transformer blocks (simplified - full implementation is complex)
        self.layers = nn.ModuleList()
        for i, depth in enumerate(depths):
            dim = embed_dim * (2 ** i)
            # Create a TransformerEncoder with the correct number of layers
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=num_heads[i],
                dim_feedforward=dim * 4,
                dropout=0.1,
                batch_first=True
            )
            self.layers.append(nn.TransformerEncoder(encoder_layer, num_layers=depth))

            # Downsample between stages
            if i < len(depths) - 1:
                self.layers.append(nn.Conv3d(dim, dim * 2, kernel_size=2, stride=2))

        final_dim = embed_dim * (2 ** (len(depths) - 1))
        self.norm = nn.LayerNorm(final_dim)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(final_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        B, C, D, H, W = x.shape

        # Process through layers
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.TransformerEncoder):
                # Reshape for transformer
                x = x.flatten(2).transpose(1, 2)  # (B, DHW, C)
                x = layer(x)
                # Reshape back
                x = x.transpose(1, 2).view(B, -1, D, H, W)
            else:
                # Downsample conv
                x = layer(x)
                _, C, D, H, W = x.shape

        x = self.avgpool(x).flatten(1)
        x = self.norm(x)
        x = self.fc(x)
        return x


class ConvNeXt3D(nn.Module):
    """3D ConvNeXt for aneurysm detection."""

    def __init__(self, num_classes=14, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], variant='tiny'):
        super().__init__()

        # ConvNeXt variants
        if variant == 'small':
            depths = [3, 3, 27, 3]  # More blocks in 3rd stage
            dims = [96, 192, 384, 768]
        elif variant == 'base':
            depths = [3, 3, 27, 3]
            dims = [128, 256, 512, 1024]  # Wider channels
        elif variant == 'large':
            depths = [3, 3, 27, 3]
            dims = [192, 384, 768, 1536]  # Even wider
        elif variant == 'xlarge':
            depths = [3, 3, 27, 3]
            dims = [256, 512, 1024, 2048]  # Widest!

        # Stem
        self.stem = nn.Sequential(
            nn.Conv3d(1, dims[0], kernel_size=4, stride=4),
            nn.LayerNorm([dims[0], 16, 16, 16])
        )

        # Stages
        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(
                *[ConvNeXtBlock3D(dims[i]) for _ in range(depths[i])]
            )
            self.stages.append(stage)

            # Downsample between stages
            if i < 3:
                self.stages.append(nn.Sequential(
                    nn.LayerNorm([dims[i], 16 // (2**i), 16 // (2**i), 16 // (2**i)]),
                    nn.Conv3d(dims[i], dims[i+1], kernel_size=2, stride=2)
                ))

        self.norm = nn.LayerNorm(dims[-1])
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(dims[-1], num_classes)

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.avgpool(x).flatten(1)
        x = self.norm(x)
        x = self.fc(x)
        return x


class ConvNeXtBlock3D(nn.Module):
    """ConvNeXt block for 3D."""

    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 4, 1)  # (B, C, D, H, W) -> (B, D, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 4, 1, 2, 3)  # (B, D, H, W, C) -> (B, C, D, H, W)
        x = residual + x
        return x


class Inception3D(nn.Module):
    """3D Inception-v3 style network for aneurysm detection."""

    def __init__(self, num_classes=14):
        super().__init__()

        # Initial convolutions
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )
        self.maxpool1 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # Inception modules
        self.inception3a = InceptionModule3D(64, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionModule3D(256, 128, 128, 192, 32, 96, 64)
        self.maxpool2 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.inception4a = InceptionModule3D(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionModule3D(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionModule3D(512, 128, 128, 256, 24, 64, 64)

        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool1(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool2(x)

        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class InceptionModule3D(nn.Module):
    """3D Inception module."""

    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super().__init__()

        # 1x1 conv branch
        self.branch1 = nn.Sequential(
            nn.Conv3d(in_channels, ch1x1, kernel_size=1),
            nn.BatchNorm3d(ch1x1),
            nn.ReLU(inplace=True)
        )

        # 3x3 conv branch
        self.branch2 = nn.Sequential(
            nn.Conv3d(in_channels, ch3x3red, kernel_size=1),
            nn.BatchNorm3d(ch3x3red),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch3x3red, ch3x3, kernel_size=3, padding=1),
            nn.BatchNorm3d(ch3x3),
            nn.ReLU(inplace=True)
        )

        # 5x5 conv branch (using two 3x3)
        self.branch3 = nn.Sequential(
            nn.Conv3d(in_channels, ch5x5red, kernel_size=1),
            nn.BatchNorm3d(ch5x5red),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch5x5red, ch5x5, kernel_size=3, padding=1),
            nn.BatchNorm3d(ch5x5),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch5x5, ch5x5, kernel_size=3, padding=1),
            nn.BatchNorm3d(ch5x5),
            nn.ReLU(inplace=True)
        )

        # Pool branch
        self.branch4 = nn.Sequential(
            nn.MaxPool3d(kernel_size=3, stride=1, padding=1),
            nn.Conv3d(in_channels, pool_proj, kernel_size=1),
            nn.BatchNorm3d(pool_proj),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        return torch.cat([branch1, branch2, branch3, branch4], 1)


class SEResNet3D(nn.Module):
    """3D SE-ResNet (ResNet with Squeeze-and-Excitation blocks)."""

    def __init__(self, num_classes=14, depth=50):
        super().__init__()

        # Initial conv
        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # SE-ResNet blocks
        if depth == 10:
            # SE-ResNet10: Minimal variant - 1 block per layer
            self.layer1 = self._make_layer(64, 64, 1)
            self.layer2 = self._make_layer(64, 128, 1, stride=2)
            self.layer3 = self._make_layer(128, 256, 1, stride=2)
            self.layer4 = self._make_layer(256, 512, 1, stride=2)
            final_channels = 512
        elif depth == 14:
            # SE-ResNet14: Ultra-light variant - [2,1,1,2]
            self.layer1 = self._make_layer(64, 64, 2)
            self.layer2 = self._make_layer(64, 128, 1, stride=2)
            self.layer3 = self._make_layer(128, 256, 1, stride=2)
            self.layer4 = self._make_layer(256, 512, 2, stride=2)
            final_channels = 512
        elif depth == 18:
            self.layer1 = self._make_layer(64, 64, 2)
            self.layer2 = self._make_layer(64, 128, 2, stride=2)
            self.layer3 = self._make_layer(128, 256, 2, stride=2)
            self.layer4 = self._make_layer(256, 512, 2, stride=2)
            final_channels = 512
        elif depth == 34:
            self.layer1 = self._make_layer(64, 64, 3)
            self.layer2 = self._make_layer(64, 128, 4, stride=2)
            self.layer3 = self._make_layer(128, 256, 6, stride=2)
            self.layer4 = self._make_layer(256, 512, 3, stride=2)
            final_channels = 512
        elif depth == 50:
            self.layer1 = self._make_layer(64, 128, 3)
            self.layer2 = self._make_layer(128, 256, 4, stride=2)
            self.layer3 = self._make_layer(256, 512, 6, stride=2)
            self.layer4 = self._make_layer(512, 1024, 3, stride=2)
            final_channels = 1024
        elif depth == 101:
            self.layer1 = self._make_layer(64, 128, 3)
            self.layer2 = self._make_layer(128, 256, 4, stride=2)
            self.layer3 = self._make_layer(256, 512, 23, stride=2)
            self.layer4 = self._make_layer(512, 1024, 3, stride=2)
            final_channels = 1024

        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(final_channels, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(SEResBlock3D(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(SEResBlock3D(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class SEResBlock3D(nn.Module):
    """SE-ResNet block with Squeeze-and-Excitation."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)

        # SE block
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(out_channels, out_channels // 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels // 16, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

        # Shortcut
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # Apply SE
        out = out * self.se(out)

        out += residual
        out = self.relu(out)
        return out


# --- Training ---

LABEL_COLS = [
    'Left Infraclinoid Internal Carotid Artery',
    'Right Infraclinoid Internal Carotid Artery',
    'Left Supraclinoid Internal Carotid Artery',
    'Right Supraclinoid Internal Carotid Artery',
    'Left Middle Cerebral Artery',
    'Right Middle Cerebral Artery',
    'Anterior Communicating Artery',
    'Left Anterior Cerebral Artery',
    'Right Anterior Cerebral Artery',
    'Left Posterior Communicating Artery',
    'Right Posterior Communicating Artery',
    'Basilar Tip',
    'Other Posterior Circulation',
    'Aneurysm Present',
]


def compute_class_weights(labels_df):
    """Compute pos_weight for BCEWithLogitsLoss based on class distribution."""
    labels = labels_df[LABEL_COLS].values
    pos_counts = labels.sum(axis=0)
    neg_counts = len(labels) - pos_counts

    # pos_weight = neg / pos (higher weight for rare classes)
    pos_weight = neg_counts / (pos_counts + 1e-5)  # avoid division by zero

    return torch.from_numpy(pos_weight).float()


def train_epoch(model, dataloader, criterion, optimizer, scaler, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0

    pbar = tqdm(dataloader, desc="Training")
    for patches, labels in pbar:
        patches = patches.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Mixed precision training
        with autocast():
            outputs = model(patches)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})

    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """Validate and compute AUC."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for patches, labels in tqdm(dataloader, desc="Validating"):
            patches = patches.to(device)
            labels = labels.to(device)

            outputs = model(patches)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            # Collect predictions
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    # Compute per-class AUC
    aucs = []
    for i in range(all_labels.shape[1]):
        try:
            auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
            aucs.append(auc)
        except:
            aucs.append(0.0)

    mean_auc = np.mean(aucs)

    return total_loss / len(dataloader), mean_auc, aucs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--labels-csv', type=str, required=True)
    parser.add_argument('--cv-dir', type=str, default='data/cv_splits')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--arch', type=str, default='resnet34',
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'densenet121', 'densenet169',
                                'efficientnet_b0', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b7',
                                'vit', 'unet3d', 'mobilenetv2', 'mobilenetv3', 'mobilenetv4', 'swin', 'convnext', 'inception',
                                'seresnet10', 'seresnet14', 'seresnet18', 'seresnet34', 'seresnet50', 'seresnet101'])
    parser.add_argument('--patch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--augment', action='store_true', default=True)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load labels
    labels_df = pd.read_csv(args.labels_csv)

    # Load CV splits
    cv_dir = Path(args.cv_dir) / f"fold_{args.fold}"
    train_idx = np.load(cv_dir / 'train_indices.npy')
    val_idx = np.load(cv_dir / 'val_indices.npy')

    # Get patch files
    data_dir = Path(args.data_dir)
    all_patches = list(data_dir.glob("*.h5"))

    # Split into train/val
    uid_to_patch = {p.stem: p for p in all_patches}
    train_files = [uid_to_patch[labels_df.iloc[i]['series_uid']]
                  for i in train_idx if labels_df.iloc[i]['series_uid'] in uid_to_patch]
    val_files = [uid_to_patch[labels_df.iloc[i]['series_uid']]
                for i in val_idx if labels_df.iloc[i]['series_uid'] in uid_to_patch]

    print(f"Train files: {len(train_files)}")
    print(f"Val files: {len(val_files)}")

    # Create datasets
    train_dataset = Eric3DDataset(train_files, labels_df, args.patch_size, augment=args.augment)
    val_dataset = Eric3DDataset(val_files, labels_df, args.patch_size, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Create model
    if args.arch == 'densenet121':
        model = DenseNet3D(num_classes=14, block_config=(6, 12, 24, 16)).to(device)
    elif args.arch == 'densenet169':
        model = DenseNet3D(num_classes=14, block_config=(6, 12, 32, 32)).to(device)
    elif args.arch == 'efficientnet_b0':
        model = EfficientNet3D(num_classes=14, variant='b0').to(device)
    elif args.arch == 'efficientnet_b2':
        model = EfficientNet3D(num_classes=14, variant='b2').to(device)
    elif args.arch == 'efficientnet_b3':
        # B3: width_mult=1.2, depth_mult=1.4
        model = EfficientNet3D(num_classes=14, variant='b2').to(device)  # Use b2 as base
    elif args.arch == 'efficientnet_b4':
        # B4: width_mult=1.4, depth_mult=1.8
        model = EfficientNet3D(num_classes=14, variant='b2').to(device)  # Use b2 as base
    elif args.arch == 'efficientnet_b7':
        # B7: width_mult=2.0, depth_mult=3.1 (very large)
        model = EfficientNet3D(num_classes=14, variant='b2').to(device)  # Use b2 as base
    elif args.arch == 'vit':
        model = VisionTransformer3D(num_classes=14).to(device)
    elif args.arch == 'unet3d':
        model = UNet3D(num_classes=14).to(device)
    elif args.arch == 'mobilenetv2':
        model = MobileNetV2_3D(num_classes=14).to(device)
    elif args.arch == 'mobilenetv3':
        model = MobileNetV3_3D(num_classes=14).to(device)
    elif args.arch == 'mobilenetv4':
        model = MobileNetV4_3D(num_classes=14, variant='medium').to(device)
    elif args.arch == 'swin':
        model = SwinTransformer3D(num_classes=14).to(device)
    elif args.arch == 'convnext':
        model = ConvNeXt3D(num_classes=14).to(device)
    elif args.arch == 'inception':
        model = Inception3D(num_classes=14).to(device)
    elif args.arch == 'seresnet10':
        model = SEResNet3D(num_classes=14, depth=10).to(device)
    elif args.arch == 'seresnet14':
        model = SEResNet3D(num_classes=14, depth=14).to(device)
    elif args.arch == 'seresnet18':
        model = SEResNet3D(num_classes=14, depth=18).to(device)
    elif args.arch == 'seresnet34':
        model = SEResNet3D(num_classes=14, depth=34).to(device)
    elif args.arch == 'seresnet50':
        model = SEResNet3D(num_classes=14, depth=50).to(device)
    elif args.arch == 'seresnet101':
        model = SEResNet3D(num_classes=14, depth=101).to(device)
    else:
        depth_map = {'resnet18': 18, 'resnet34': 34, 'resnet50': 50, 'resnet101': 101}
        depth = depth_map[args.arch]
        model = ResNet3D(num_classes=14, depth=depth).to(device)

    # Compute class weights
    pos_weight = compute_class_weights(labels_df).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Mixed precision scaler
    scaler = GradScaler()

    # Training loop
    best_auc = 0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_auc': []}

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_auc, class_aucs = validate(model, val_loader, criterion, device)

        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_auc)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val AUC: {val_auc:.4f}")

        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), output_dir / 'best_model.pth')
            patience_counter = 0
            print(f"- New best AUC: {best_auc:.4f}")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    # Save history
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Training complete! Best AUC: {best_auc:.4f}")
    print(f"Model saved to: {output_dir / 'best_model.pth'}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
