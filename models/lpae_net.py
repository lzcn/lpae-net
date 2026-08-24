"""LPAE-Net: Learnable Prototype Anchor Embedding Network (CVPR 2021).

Two model variants are provided:

- :class:`LPAENet`: user-specific learnable anchor prototypes in a memory
  module (``lpae_u``) optionally combined with shared anchors (``lpae_g``).
- :class:`LatentFactorNet`: user embeddings matched to outfit embeddings.
"""

import math
from dataclasses import asdict, dataclass, field
from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn

from models.backbones import build_backbone


def one_hot(index: torch.Tensor, num: int) -> torch.Tensor:
    """One-hot encoding on the same device as ``index``."""
    index = index.view(-1, 1)
    out = torch.zeros(index.numel(), num, device=index.device)
    return out.scatter_(1, index, 1.0)


@dataclass
class BaseParam:
    @classmethod
    def from_dict(cls, cfg: Dict):
        """Create a param instance from a dict, ignoring unknown keys."""
        valid = set(asdict(cls()).keys())
        return cls(**{k: v for k, v in dict(cfg).items() if k in valid})


@dataclass
class LPAENetParam(BaseParam):
    name: str = "LPAENet"
    num_users: int = 630
    backbone: str = "resnet34"
    embd_dim: int = 128
    com_memory: bool = False
    loss_weight: Dict[str, float] = field(default_factory=lambda: {"rank_loss": 1.0})
    num_points: int = 0  # number of inducing points; 0 -> stacked SABs
    num_proto: int = 16
    num_sab: int = 2
    num_seeds: int = 1
    num_heads: int = 4
    logdet: bool = True
    use_nn_feature: bool = False
    use_semantic: bool = False
    use_visual: bool = False
    cold_start: bool = False


@dataclass
class LatentFactorNetParam(BaseParam):
    name: str = "LatentFactorNet"
    backbone: str = "resnet34"
    embd_dim: int = 128
    num_users: int = 630
    loss_weight: Dict[str, float] = field(default_factory=lambda: {"rank_loss": 1.0})
    num_points: int = 0
    num_sab: int = 2
    num_seeds: int = 1
    com_score: bool = False
    num_heads: int = 4
    use_nn_feature: bool = False
    use_semantic: bool = False
    use_visual: bool = False
    cold_start: bool = False


class MAB(nn.Module):
    """Multi-head attention block."""

    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super().__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K, mask_a, mask_b):
        # b x n x d
        Q, K, V = self.fc_q(Q), self.fc_k(K), self.fc_v(K)
        dim_split = self.dim_V // self.num_heads
        # (h b) x n x 1
        mask_a = mask_a.repeat(self.num_heads, 1, 1)
        mask_b = mask_b.repeat(self.num_heads, 1, 1)
        # (h b) x n x d
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)
        # (h b) x n x n
        dots = Q_.bmm(K_.transpose(1, 2)) / math.sqrt(dim_split)
        mask = torch.bmm(mask_a, mask_b.transpose(1, 2)) == 1.0
        dots.masked_fill_(~mask, -torch.finfo(dots.dtype).max)
        A = torch.softmax(dots, dim=2)
        # (h b) x n x d -> b x n x (h d)
        H = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        H = H if getattr(self, "ln0", None) is None else self.ln0(H)
        H = H + F.relu(self.fc_o(H))
        H = H if getattr(self, "ln1", None) is None else self.ln1(H)
        return H


class SAB(nn.Module):
    """Self-attention block."""

    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super().__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X, mask):
        return self.mab(X, X, mask, mask)


class ISAB(nn.Module):
    """Inducing point self-attention block."""

    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super().__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))  # noqa: E741
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X, mask):
        # inducing points are always valid
        i_mask = torch.ones(X.size(0), self.I.size(1), 1, device=X.device, dtype=X.dtype)
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X, i_mask, mask)
        return self.mab1(X, H, mask, i_mask)


class PMA(nn.Module):
    """Pooling by multi-head attention."""

    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super().__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        self.register_buffer("mask", torch.ones(1, num_seeds, 1))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X, mask):
        b = X.size(0)
        return self.mab(self.S.repeat(b, 1, 1), X, self.mask.repeat(b, 1, 1), mask)


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


def stack_sab(dim, num_heads, num_sab, num_points=0):
    def _isab():
        return ISAB(dim, dim, num_heads, num_points, ln=True)

    def _sab():
        return SAB(dim, dim, num_heads, ln=True)

    sab = _isab if num_points and num_points > 0 else _sab
    return nn.ModuleList([sab() for _ in range(num_sab)])


def embed_layer(in_features, out_features, dropout=0.2) -> nn.Module:
    x = nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.LayerNorm(out_features),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
    )
    x.apply(weights_init)
    return x


class SetTransformer(nn.Module):
    """Encode a set of item features into ``num_seeds`` outfit embeddings."""

    def __init__(
        self,
        in_features,
        embd_dim=128,
        num_sab=2,
        num_points=16,
        num_heads=4,
        num_seeds=4,
    ):
        super().__init__()
        self.embed = embed_layer(in_features, embd_dim)
        self.encoder = stack_sab(dim=embd_dim, num_heads=num_heads, num_sab=num_sab, num_points=num_points)
        self.pma = PMA(embd_dim, num_heads, num_seeds, ln=True)
        self.sab = SAB(embd_dim, embd_dim, num_heads, ln=True) if num_seeds > 1 else nn.Identity()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: a float tensor with shape [n, b, d].
            mask: a float tensor with shape [b, n, 1], 1 for valid items.
        Returns:
            an output tensor with shape [num_seeds, b, d].
        """
        h = self.embed(x)
        for encoder in self.encoder:
            h = encoder(h, mask)
        z = h
        # h = SAB(PMA(k, rFF(Z))), shape [b, k, d]
        out = self.sab(self.pma(z, mask))
        return out.permute(1, 0, 2)


class UserEmbedding(nn.Module):
    """Learnable one-hot user embedding."""

    def __init__(self, num_users, dim):
        super().__init__()
        self.num_users = num_users
        self.encoder = nn.Linear(num_users, dim, bias=False)
        nn.init.normal_(self.encoder.weight, std=0.01)

    def forward(self, x):
        return self.encoder(one_hot(x, self.num_users))


class UserMemory(nn.Module):
    """Neighborhood-based prototype memory network."""

    def __init__(self, num_users, embd_dim, num_proto, com_memory=False, logdet=True):
        super().__init__()
        self.d = embd_dim
        self.logdet = logdet
        self.num_users = num_users
        self.num_proto = num_proto
        self.com_memory = com_memory
        self.users = nn.Parameter(torch.zeros(num_users, 1, embd_dim))
        self.trans = nn.Sequential(nn.Linear(embd_dim, embd_dim))
        if com_memory:
            self.out_proto = nn.Parameter(torch.zeros(num_users, num_proto // 2, embd_dim))
            self.com_proto = nn.Parameter(torch.zeros(1, num_proto // 2, embd_dim))
        else:
            self.out_proto = nn.Parameter(torch.zeros(num_users, num_proto, embd_dim))
            self.com_proto = None
        self.init()

    def init(self):
        scale = math.sqrt(3 / self.d)
        nn.init.uniform_(self.out_proto, a=-scale, b=scale)
        nn.init.uniform_(self.users, a=-scale, b=scale)
        if self.com_memory:
            nn.init.uniform_(self.com_proto, a=-scale, b=scale)
        for m in self.trans.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=math.sqrt(2 / self.d))

    def reg(self, uidx: torch.Tensor):
        """Log-determinant regularizer over the prototypes of each user."""
        out_proto = self.out_proto.index_select(0, uidx)
        s = out_proto.size(1)
        n = uidx.size(0)
        g = out_proto.matmul(out_proto.transpose(1, 2))
        eye = torch.eye(s, device=g.device, dtype=g.dtype)
        if self.logdet:
            reg = (g.diagonal(dim1=1, dim2=2).sum() - torch.logdet(g).sum()) / n
        # orthogonality regularization as an alternative when the
        # log-determinant raises illegal GPU memory access
        else:
            reg = ((g * (torch.ones_like(g) - eye)) ** 2).sum() / n
        return reg

    def forward(self, uidx: torch.Tensor, x: torch.Tensor, attn_map=False):
        n = x.size(0)
        x = x.unsqueeze(1)
        # n x s x d
        if self.com_memory:
            u_anchors = self.out_proto.index_select(0, uidx)
            g_anchors = self.com_proto.repeat(n, 1, 1)
            anchors = torch.cat((u_anchors, g_anchors), dim=1)
        else:
            anchors = self.out_proto.index_select(0, uidx)
        # cosine similarity
        h = self.trans(x)
        z = F.normalize(h, dim=-1)
        anchors = F.normalize(anchors, dim=-1)
        # n x 1 x s
        cos_sim = z.matmul(anchors.transpose(1, 2))
        # n x 1
        score = cos_sim.sum(dim=-1).view(-1, 1)
        if attn_map:
            return score, cos_sim
        return score


class LPAENet(nn.Module):
    """LPAE-Net."""

    def __init__(self, param: LPAENetParam):
        super().__init__()
        self.param = param
        if param.use_visual:
            backbone, in_features = build_backbone(param.backbone)
            # pre-extracted features are used directly when use_nn_feature
            self.visual_feature = nn.Identity() if param.use_nn_feature else backbone
            self.visual_encoder = SetTransformer(
                in_features=in_features,
                embd_dim=param.embd_dim,
                num_sab=param.num_sab,
                num_points=param.num_points,
                num_seeds=param.num_seeds,
            )
        if param.use_semantic:
            raise NotImplementedError("Semantic features are not supported in this release.")
        self.memory = UserMemory(
            num_users=param.num_users,
            embd_dim=param.embd_dim,
            num_proto=param.num_proto,
            com_memory=param.com_memory,
            logdet=param.logdet,
        )
        if param.cold_start:
            self.cold_start()

    def cold_start(self):
        for param in self.parameters():
            param.requires_grad = False
        self.memory.out_proto.requires_grad = True

    @torch.no_grad()
    def test_batch(self, data: torch.Tensor, uidx: torch.Tensor, cate: torch.Tensor):
        """Test forward."""
        batch, num, *shape = data.shape
        data = data.view(-1, *shape)
        feat = self.visual_feature(data)
        feat = feat.view(batch, num, -1)
        mask = 1.0 * (cate.view(batch, num, -1) != -1)
        feat = self.visual_encoder(feat, mask).reshape(-1, self.param.embd_dim)
        scores = self.memory(uidx, feat)
        return scores

    def train_batch(self, data: torch.Tensor, uidx: torch.Tensor, cate: torch.Tensor):
        """Training forward."""
        batch, _, num, *shape = data.shape
        data = data.view(-1, *shape)
        feat = self.visual_feature(data)
        feat = feat.view(batch * 2, num, -1)
        mask = 1.0 * (cate.view(batch * 2, num, -1) != -1)
        feat = self.visual_encoder(feat, mask).reshape(-1, self.param.embd_dim)
        feat = feat.view(batch, 2, -1)
        pos_feat, neg_feat = feat.split(1, dim=1)
        pos_score = self.memory(uidx, pos_feat.squeeze())
        neg_score = self.memory(uidx, neg_feat.squeeze())
        diff = pos_score.view(-1, 1) - neg_score.view(1, -1)
        diff = diff.view(-1)
        rank = F.soft_margin_loss(diff, torch.ones_like(diff), reduction="none")
        loss = dict(rank_loss=rank, l1reg=self.memory.reg(uidx))
        accuracy = dict(accuracy=torch.gt(diff, 0.0))
        return loss, accuracy

    def forward(self, *inputs):
        if self.training:
            return self.train_batch(*inputs)
        return self.test_batch(*inputs)


class LatentMatch(nn.Module):
    def __init__(self, embd_dim, com_score=False):
        super().__init__()
        self.d = d = embd_dim
        self.com_score = com_score
        self.dense_g = nn.Parameter(torch.zeros(d, d))
        self.out_g = nn.Parameter(torch.zeros(d, 1))
        self.bias_g = nn.Parameter(torch.zeros(1, d))
        if com_score:
            self.dense_c = nn.Parameter(torch.zeros(d, d))
            self.out_c = nn.Parameter(torch.zeros(d, 1))
            self.bias_c = nn.Parameter(torch.zeros(1, d))
            self.scale = nn.Parameter(torch.ones(1, 1))
        self.init()

    def init(self):
        nn.init.normal_(self.out_g, std=math.sqrt(2 / self.d))
        nn.init.normal_(self.dense_g, std=math.sqrt(2 / self.d))
        if self.com_score:
            nn.init.normal_(self.out_c, std=math.sqrt(2 / self.d))
            nn.init.normal_(self.dense_c, std=math.sqrt(2 / self.d))

    def forward(self, u, z):
        u = F.normalize(u, dim=-1)
        z = F.normalize(z, dim=-1)
        if self.com_score:
            score_u = F.relu((u * z).matmul(self.dense_g) + self.bias_g).matmul(self.out_g)
            score_i = F.relu(z.matmul(self.dense_c) + self.bias_c).matmul(self.out_c)
            score = score_u * self.scale + score_i
        else:
            score = F.relu((u * z).matmul(self.dense_g) + self.bias_g).matmul(self.out_g)
        return score


class LatentFactorNet(nn.Module):
    """Basic outfit transformer with user embeddings.

    Args:
        embd_dim: embedding dimension
        num_sab: number of SABs or iSABs
        num_points: number of induced points; 0 -> stacked SABs
        num_seeds: number of seed vectors for outputs
    """

    def __init__(self, param: LatentFactorNetParam):
        super().__init__()
        self.param = param
        if param.use_visual:
            backbone, in_features = build_backbone(param.backbone)
            self.visual_feature = nn.Identity() if param.use_nn_feature else backbone
            self.visual_encoder = SetTransformer(
                in_features=in_features,
                embd_dim=param.embd_dim,
                num_sab=param.num_sab,
                num_points=param.num_points,
                num_seeds=param.num_seeds,
                num_heads=param.num_heads,
            )
        if param.use_semantic:
            raise NotImplementedError("Semantic features are not supported in this release.")
        self.users = UserEmbedding(param.num_users, param.embd_dim)
        self.match = LatentMatch(param.embd_dim, param.com_score)
        if param.cold_start:
            self.cold_start()

    def cold_start(self):
        for param in self.parameters():
            param.requires_grad = False
        self.users.encoder.weight.requires_grad = True

    @torch.no_grad()
    def test_batch(self, data: torch.Tensor, uidx: torch.Tensor, cate: torch.Tensor):
        batch, num, *shape = data.shape
        data = data.view(-1, *shape)
        feat = self.visual_feature(data)
        feat = feat.view(batch, num, -1)
        mask = 1.0 * (cate.view(batch, num, -1) != -1)
        feat = self.visual_encoder(feat, mask).reshape(-1, self.param.embd_dim)
        users = self.users(uidx)
        return self.match(users, feat)

    def train_batch(self, data: torch.Tensor, uidx: torch.Tensor, cate: torch.Tensor):
        users = self.users(uidx)
        batch, _, num, *shape = data.shape
        data = data.view(-1, *shape)
        feat = self.visual_feature(data)
        feat = feat.view(batch * 2, num, -1)
        mask = 1.0 * (cate.view(batch * 2, num, -1) != -1)
        feat = self.visual_encoder(feat, mask).reshape(-1, self.param.embd_dim)
        feat = feat.view(batch, 2, -1)
        pos_feat, neg_feat = feat.split(1, dim=1)
        pos_score = self.match(users, pos_feat.squeeze())
        neg_score = self.match(users, neg_feat.squeeze())
        diff = pos_score - neg_score
        rank = F.soft_margin_loss(diff, torch.ones_like(diff), reduction="none")
        loss = dict(rank_loss=rank)
        accuracy = dict(accuracy=torch.gt(diff, 0.0))
        return loss, accuracy

    def forward(self, *inputs):
        if self.training:
            return self.train_batch(*inputs)
        return self.test_batch(*inputs)
