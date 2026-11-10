import math
import open_clip
import torch
from torch import nn
from torch.nn.utils import parametrize


class QVLoRA(nn.Module):
    def __init__(self, width, rank=8, alpha=8):
        super().__init__()
        self.scale = alpha / rank
        self.q_down = nn.Parameter(torch.empty(rank, width))
        self.q_up = nn.Parameter(torch.zeros(width, rank))
        self.v_down = nn.Parameter(torch.empty(rank, width))
        self.v_up = nn.Parameter(torch.zeros(width, rank))
        nn.init.kaiming_uniform_(self.q_down, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.v_down, a=math.sqrt(5))

    def forward(self, weight):
        width = weight.shape[1]
        delta = torch.cat((
            self.q_up @ self.q_down,
            torch.zeros_like(weight[width:2 * width]),
            self.v_up @ self.v_down
        ), dim=0)
        return weight + self.scale * delta


class CLIPEncoder(nn.Module):
    def __init__(self, pretrained=True, lora=False, rank=8, alpha=8):
        super().__init__()
        clip = open_clip.create_model("ViT-B-16", pretrained="openai" if pretrained else None)
        self.visual = clip.visual
        del clip

        self.channels = int(self.visual.proj.shape[-1])
        self.lora = lora

        if lora:
            for p in self.visual.parameters():
                p.requires_grad_(False)

            for block in self.visual.transformer.resblocks:
                attention = block.attn
                parametrize.register_parametrization(
                    attention, "in_proj_weight", QVLoRA(768, rank, alpha)
                )

    def forward(self, x):
        v = self.visual
        x = v.conv1(x)
        b, c, h, w = x.shape
        x = x.reshape(b, c, h * w).permute(0, 2, 1)
        cls = v.class_embedding.to(x.dtype).expand(b, 1, -1)
        x = torch.cat((cls, x), dim=1)
        x = x + v.positional_embedding.to(x.dtype)
        x = v.patch_dropout(x)
        x = v.ln_pre(x)
        x = v.transformer(x)
        x = v.ln_post(x)
        x = x @ v.proj
        return x[:, 1:], x[:, 0], (h, w)
