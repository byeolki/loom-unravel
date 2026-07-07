from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

NUM_LANDMARKS = 24
INPUT_SIZE = 128
VGG16_WEIGHTS_URL = "https://download.pytorch.org/models/vgg16-397923af.pth"


class CascadedFaceAlignment(nn.Module):
    def __init__(self, output_channel_num: int = NUM_LANDMARKS + 1, checkpoint_path: str | None = None):
        super().__init__()

        self.output_channel_num = output_channel_num
        self.stage_channel_num = 128
        self.stage_num = 2

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
        )

        self.CFM_features = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, self.stage_channel_num, kernel_size=3, padding=1), nn.ReLU(inplace=True),
        )

        stages = [self._make_stage(self.stage_channel_num)]
        for _ in range(1, self.stage_num):
            stages.append(self._make_stage(self.stage_channel_num + self.output_channel_num))
        self.stages = nn.ModuleList(stages)

        if checkpoint_path:
            snapshot = torch.load(checkpoint_path, map_location="cpu")
            self.load_state_dict(snapshot["state_dict"])
        else:
            self._load_vgg16_backbone()

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        feature = self.features(x)
        feature = self.CFM_features(feature)
        heatmaps = [self.stages[0](feature)]
        for i in range(1, self.stage_num):
            heatmaps.append(self.stages[i](torch.cat([feature, heatmaps[i - 1]], dim=1)))
        return heatmaps

    def _make_stage(self, in_channels: int) -> nn.Sequential:
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, self.stage_channel_num, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ]
        for _ in range(4):
            layers.append(nn.Conv2d(self.stage_channel_num, self.stage_channel_num, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(self.stage_channel_num, self.output_channel_num, kernel_size=3, padding=1))
        return nn.Sequential(*layers)

    def _load_vgg16_backbone(self) -> None:
        vgg16_state_dict = model_zoo.load_url(VGG16_WEIGHTS_URL)
        own_state_dict = self.state_dict()
        merged = OrderedDict(
            (key, vgg16_state_dict[key] if key in vgg16_state_dict else value)
            for key, value in own_state_dict.items()
        )
        self.load_state_dict(merged)
