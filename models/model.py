
import copy
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .clip_encoder import CLIPEncoder
from .foundation_models import create_clip_lora


class PADModel(nn.Module):
    def __init__(self, backbone="clip_vitb16_lora", dim=192, pretrained=True, lora_rank=8, lora_alpha=8):
        super().__init__()
        self.is_clip = True
        self.encoder = CLIPEncoder(pretrained=pretrained, lora=backbone.endswith("_lora"),
                                   rank=lora_rank, alpha=lora_alpha)
        self.channels = self.encoder.channels
        self.projection = nn.Linear(self.channels, dim)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 2)
        self.dim = dim

    def features(self, x):
        patches, cls, shape = self.encoder(x)
        z = self.norm(self.projection(patches))
        global_feature = self.norm(self.projection(cls))
        return z, shape, global_feature

    def forward(self, x, return_features=False):
        z, shape, pooled = self.features(x)
        logits = self.head(pooled)
        return (logits, z, shape) if return_features else logits


def position_encoding(h, w, dim, device):
    yy, xx = torch.meshgrid(torch.arange(h, device=device, dtype=torch.float32),
                            torch.arange(w, device=device, dtype=torch.float32), indexing="ij")
    omega = 1. / (10000 ** (torch.arange(dim // 4, device=device) / max(1, dim // 4)))
    x, y = xx.flatten()[:, None] * omega, yy.flatten()[:, None] * omega
    return torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), -1)[None]


class Predictor(nn.Module):
    def __init__(self, dim=192, depth=2, heads=4):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.mask_token, std=.02)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout=0.,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.net = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.output = nn.Linear(dim, dim)

    def forward(self, z, mask, shape):
        z = torch.where(mask[..., None], self.mask_token.to(z.dtype), z)
        pos = position_encoding(*shape, z.shape[-1], z.device).to(z.dtype)
        return self.output(self.net(z + pos))


@torch.no_grad()
def update_teacher(teacher, student, momentum):
    for target, source in zip(teacher.parameters(), student.parameters()):
        target.mul_(momentum).add_(source, alpha=1 - momentum)
    for target, source in zip(teacher.buffers(), student.buffers()):
        if target.is_floating_point():
            target.mul_(momentum).add_(source, alpha=1 - momentum)
        else:
            target.copy_(source)
    teacher.eval()


@torch.no_grad()
def select_mask(z1, ratio=.30):
    b, n, _ = z1.shape
    k = max(1, min(n - 1, round(n * ratio)))
    score = torch.rand(b, n, device=z1.device)
    mask = torch.zeros(b, n, dtype=torch.bool, device=z1.device)
    mask.scatter_(1, score.topk(k, dim=1).indices, True)
    return mask


def mask_image(x, mask, shape):
    patch = mask.reshape(x.shape[0], 1, *shape).float()
    pixel_mask = F.interpolate(patch, size=x.shape[-2:], mode="nearest").bool()
    return x.masked_fill(pixel_mask, 0)


class CAPETrainerModel(nn.Module):
    def __init__(self, student, mode="random"):
        super().__init__()
        self.student = student
        self.predictor = Predictor(student.dim)
        self.teacher = copy.deepcopy(student).eval().requires_grad_(False)
        self.mode = mode

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        return self

    def losses(self, view1, view2, labels, ratio=.3, active=True,
               prediction_weight=1., masked_weight=.5, consistency_weight=.1):
        logits = self.student(view2)
        ce = F.cross_entropy(logits.float(), labels)
        zero = ce.new_zeros(())
        if not active or self.mode in ("baseline", "augment"):
            return {"total": ce, "classification": ce, "prediction": zero,
                    "masked_classification": zero, "consistency": zero}

        with torch.no_grad():
            _, target1, shape = self.teacher(view1, True)
            mask = select_mask(target1, ratio)
            target = F.normalize(target1.float(), dim=-1)

        masked = mask_image(view2, mask, shape)
        masked_logits, context, _ = self.student(masked, True)
        pred = self.predictor(context, mask, shape)
        prediction = (1 - (F.normalize(pred.float(), dim=-1) * target).sum(-1))[mask].mean()
        masked_ce = F.cross_entropy(masked_logits.float(), labels)
        consistency = F.kl_div(F.log_softmax(masked_logits.float(), -1),
                               F.softmax(logits.detach().float(), -1), reduction="batchmean")
        total = ce + prediction_weight * prediction + masked_weight * masked_ce + consistency_weight * consistency
        return {"total": total, "classification": ce, "prediction": prediction,
                "masked_classification": masked_ce, "consistency": consistency}


def load_sources(project):
    return create_clip_lora, CAPETrainerModel, update_teacher


class CLIPStudent(nn.Module):
    def __init__(self, original, dim=192):
        super().__init__()
        self.original = original
        self.dim = dim
        self.projection = nn.Linear(original.backbone.output_dim, dim)
        self.head = original.classifier

    def forward(self, x, return_features=False):
        if not return_features:
            return self.original(x)

        captured = []
        handle = self.original.backbone.transformer.register_forward_hook(
            lambda module, inputs, output: captured.append(output))

        logits = self.original(x)
        handle.remove()

        tokens = captured[0].permute(1, 0, 2)[:, 1:, :]
        n = tokens.shape[1]
        side = int(n ** 0.5)
        features = self.original.backbone.ln_post(tokens) @ self.original.backbone.proj
        return logits, self.projection(features), (side, side)


def build_model(project, checkpoint, dim=192, mode="random"):
    original = create_clip_lora(pretrained=True)

    if checkpoint is not None:
        state = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "student" in state:
            state = state["student"]
        original.load_state_dict(state, strict=True)

    student = CLIPStudent(original, dim)
    trainer = CAPETrainerModel(student, mode)
    return trainer, update_teacher
