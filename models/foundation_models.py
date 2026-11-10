import clip
import torch
from torch import nn
from torch.nn.utils import parametrize

class CLIPPAD(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        model, _ = clip.load("ViT-B/16", device="cpu", jit=False)
        self.backbone = model.visual.float()
        self.classifier = nn.Linear(self.backbone.output_dim, 2)

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

class AttentionLoRA(nn.Module):
    def __init__(self, width, rank=8, alpha=8):
        super().__init__()
        self.scale = alpha / rank
        self.q_down = nn.Parameter(torch.empty(rank, width))
        self.q_up = nn.Parameter(torch.zeros(width, rank))
        self.v_down = nn.Parameter(torch.empty(rank, width))
        self.v_up = nn.Parameter(torch.zeros(width, rank))
        nn.init.kaiming_uniform_(self.q_down, a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.v_down, a=5 ** 0.5)

    def forward(self, weight):
        width = weight.shape[1]
        update = torch.zeros_like(weight)
        update[:width] = self.q_up @ self.q_down
        update[2 * width:] = self.v_up @ self.v_down
        return weight + self.scale * update

class CLIPLoRAPAD(CLIPPAD):
    def __init__(self, pretrained=True):
        super().__init__(pretrained=pretrained)

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        for block in self.backbone.transformer.resblocks:
            attn = block.attn
            parametrize.register_parametrization(attn, "in_proj_weight", AttentionLoRA(attn.embed_dim))

def create_clip_lora(pretrained=True):
    return CLIPLoRAPAD(pretrained=pretrained)
