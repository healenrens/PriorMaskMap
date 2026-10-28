import torch
import torch.nn as nn
from mmdet.models import HEADS


class VectorMapHybridPriorEmbedding(nn.Module):
    def __init__(self, point_dim, embed_dim, num_types):
        super(VectorMapHybridPriorEmbedding, self).__init__()
        self.point_dim = point_dim
        self.embed_dim = embed_dim
        self.num_types = num_types
        self.point_embed = nn.Linear(point_dim, embed_dim)
        self.type_embed = nn.Embedding(num_types + 1, embed_dim)
        self.instance_embed = nn.Linear(1, embed_dim)
        self.position_embed = nn.Linear(2, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, points, instance_ids, type_ids, positions):
        point_emb = self.point_embed(points)
        type_emb = self.type_embed(type_ids.long())
        instance_emb = self.instance_embed(instance_ids.unsqueeze(-1).to(torch.float32))
        pos_emb = self.position_embed(positions)
        combined_emb = point_emb + type_emb + instance_emb + pos_emb
        return self.layer_norm(combined_emb)


class VectorMapIntraVectorAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(VectorMapIntraVectorAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        bs, num_instance, num_pts, embed_dim = x.shape
        x_reshaped = x.view(bs * num_instance, num_pts, embed_dim)
        attn_out, _ = self.attention(x_reshaped, x_reshaped, x_reshaped)
        x_reshaped = self.norm(x_reshaped + attn_out)
        return x_reshaped.view(bs, num_instance, num_pts, embed_dim)


class VectorMapInterVectorAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(VectorMapInterVectorAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        bs, num_instance, num_pts, embed_dim = x.shape
        instance_repr = x.mean(dim=2)
        attn_out, _ = self.attention(instance_repr, instance_repr, instance_repr)
        instance_repr = self.norm(instance_repr + attn_out)
        instance_repr = instance_repr.unsqueeze(2).expand(-1, -1, num_pts, -1)
        return self.norm(x + instance_repr)


class VectorMapInstanceAggregation(nn.Module):
    def __init__(self, embed_dim):
        super(VectorMapInstanceAggregation, self).__init__()
        self.embed_dim = embed_dim

    def forward(self, x, instance_ids):
        bs, num_instance, num_pts, embed_dim = x.shape
        instance_features = x.mean(dim=2)
        return instance_features


@HEADS.register_module()
class PriorVectorEncoder(nn.Module):
    """Encode a vector-prior tensor with shape [B, N, P, 3]."""

    def __init__(
        self,
        point_dim=2,
        embed_dim=64,
        num_heads=8,
        M=3,
        N=2,
        num_types=3,
        target_embed_dim=256,
        enable_reconstruction=False,
    ):
        super(PriorVectorEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.target_embed_dim = target_embed_dim
        self.enable_reconstruction = enable_reconstruction
        self.hpe = VectorMapHybridPriorEmbedding(point_dim, embed_dim, num_types)
        self.intra_vector_encoders = nn.ModuleList(
            [VectorMapIntraVectorAttention(embed_dim, num_heads) for _ in range(M)]
        )
        self.inter_vector_encoders = nn.ModuleList(
            [VectorMapInterVectorAttention(embed_dim, num_heads) for _ in range(N)]
        )
        self.instance_aggregation = VectorMapInstanceAggregation(embed_dim)
        self.position_encoder = nn.Linear(2, embed_dim)
        self.target_projection = nn.Linear(embed_dim, target_embed_dim)
        if self.enable_reconstruction:
            self.reconstruction_head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2),
                nn.ReLU(),
                nn.Linear(embed_dim * 2, embed_dim // 2),
                nn.ReLU(),
                nn.Linear(embed_dim // 2, point_dim),
            )

    def forward(self, vector_maps):
        bs, num_instance, num_pts, _ = vector_maps.shape
        points = vector_maps[..., :2]
        cls_ids = vector_maps[..., 2]
        instance_ids = (
            torch.arange(num_instance, device=vector_maps.device, dtype=torch.int64)
            .unsqueeze(0)
            .unsqueeze(-1)
            .expand(bs, -1, num_pts)
        )
        positions = points.clone()
        x = self.hpe(points, instance_ids, cls_ids, positions)
        for encoder in self.intra_vector_encoders:
            x = encoder(x)
        for encoder in self.inter_vector_encoders:
            x = encoder(x)
        instance_features = self.instance_aggregation(x, instance_ids)
        instance_features_expanded = instance_features.unsqueeze(2).expand(
            -1, -1, num_pts, -1
        )
        point_features_enhanced = x + instance_features_expanded
        position_embeds = self.position_encoder(points)
        point_features_final = point_features_enhanced + position_embeds
        point_features_projected = self.target_projection(point_features_final)
        if self.enable_reconstruction:
            reconstructed_points = self.reconstruction_head(point_features_final)
            return (point_features_projected, reconstructed_points)
        else:
            return point_features_projected
