import open_clip
import torch
from torch import nn
from torch.nn.utils import parametrize
from .clip_encoder import QVLoRA
from .model import PADModel, CAPETrainerModel

class SequentialPAD(PADModel):
    def __init__(self, dim=192, rank=8, alpha=8):
        super().__init__("clip_vitb16_lora", dim, pretrained=False, lora_rank=rank, lora_alpha=alpha)

        visual = open_clip.create_model("ViT-B-16-quickgelu", pretrained=None).visual
        for parameter in visual.parameters():
            parameter.requires_grad_(False)
        for block in visual.transformer.resblocks:
            parametrize.register_parametrization(block.attn, "in_proj_weight", QVLoRA(768, rank, alpha))

        self.encoder.visual = visual
        self.head = nn.Linear(self.channels, 2)
        self.aux_head = nn.Linear(dim, 2)

    def features(self, x):
        patches, cls, shape = self.encoder(x)
        tokens = self.norm(self.projection(patches))
        return tokens, shape, cls

    def forward(self, x, return_features=False):
        tokens, shape, cls = self.features(x)
        logits = self.head(cls)
        return (logits, tokens, shape) if return_features else logits


def load_original(student, checkpoint):
    source = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if "student" in source:
        source = source["student"]

    original_backbone = {k: v for k, v in source.items() if k.startswith("backbone.")}
    mapped = {"encoder.visual." + k[len("backbone."):]: v for k, v in original_backbone.items()}
    mapped["head.weight"] = source["classifier.weight"]
    mapped["head.bias"] = source["classifier.bias"]

    student.load_state_dict(mapped, strict=False)
    return student


class SequentialTrainer(CAPETrainerModel):
    pass
