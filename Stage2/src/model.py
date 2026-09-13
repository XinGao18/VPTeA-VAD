from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn
from clip import clip
from utils.layers import GraphConvolution, DistanceAdj

class LayerNorm(nn.LayerNorm):

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


def build_padding_mask(lengths, max_length, device):
    lengths = lengths.to(device=device, dtype=torch.long).reshape(-1)
    lengths = lengths.clamp(min=1, max=max_length)
    positions = torch.arange(max_length, device=device)
    return positions.unsqueeze(0) >= lengths.unsqueeze(1)


def load_compatible_state_dict(model, state_dict):
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    allowed_missing = {'cross_attention_alpha'} if model.use_prompt_pairs else set()
    unexpected_keys = set(unexpected)
    missing_keys = set(missing) - allowed_missing
    if unexpected_keys or missing_keys:
        raise RuntimeError(
            f'Incompatible model state dict: missing={sorted(missing_keys)}, '
            f'unexpected={sorted(unexpected_keys)}'
        )
    return missing, unexpected


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, padding_mask: torch.Tensor):
        padding_mask = padding_mask.to(dtype=bool, device=x.device) if padding_mask is not None else None
        self.attn_mask = self.attn_mask.to(device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, key_padding_mask=padding_mask, attn_mask=self.attn_mask)[0]

    def forward(self, x):
        x, padding_mask = x
        x = x + self.attention(self.ln_1(x), padding_mask)
        x = x + self.mlp(self.ln_2(x))
        return (x, padding_mask)


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VPTeAVAD(nn.Module):
    def __init__(self,
                 num_class: int,
                 embed_dim: int,
                 visual_length: int,
                 visual_width: int,
                 visual_head: int,
                 visual_layers: int,
                 attn_window: int,
                 device,
                 use_prompt_pairs: bool = False):
        super().__init__()

        self.num_class = num_class
        self.visual_length = visual_length
        self.visual_width = visual_width
        self.embed_dim = embed_dim
        self.attn_window = attn_window
        self.device = device
        self.use_prompt_pairs = use_prompt_pairs

        self.temporal = Transformer(
            width=visual_width,
            layers=visual_layers,
            heads=visual_head,
            attn_mask=self.build_attention_mask(self.attn_window)
        )

        width = int(visual_width / 2)
        self.gc1 = GraphConvolution(visual_width, width, residual=True)
        self.gc2 = GraphConvolution(width, width, residual=True)
        self.gc3 = GraphConvolution(visual_width, width, residual=True)
        self.gc4 = GraphConvolution(width, width, residual=True)
        self.disAdj = DistanceAdj()
        self.linear = nn.Linear(visual_width, visual_width)
        self.gelu = QuickGELU()

        self.mlp1 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(visual_width, visual_width * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(visual_width * 4, visual_width))
        ]))
        self.mlp2 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(visual_width, visual_width * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(visual_width * 4, visual_width))
        ]))
        self.classifier = nn.Linear(visual_width, 1)

        self.clipmodel, _ = clip.load("ViT-B/16", device)
        for clip_param in self.clipmodel.parameters():
            clip_param.requires_grad = False

        self.frame_position_embeddings = nn.Embedding(visual_length, visual_width)
        if use_prompt_pairs:
            self.visual_alignment = nn.Sequential(
                nn.Linear(visual_width, visual_width),
                LayerNorm(visual_width)
            )
            self.text_alignment = nn.Sequential(
                nn.Linear(embed_dim, visual_width),
                LayerNorm(visual_width)
            )
            self.pair_cross_attention = nn.MultiheadAttention(visual_width, visual_head)
            self.cross_attention_alpha = nn.Parameter(torch.tensor(-4.5951198501))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.frame_position_embeddings.weight, std=0.01)

    def get_cross_attention_alpha(self):
        if not self.use_prompt_pairs:
            return self.frame_position_embeddings.weight.new_zeros(())
        return torch.sigmoid(self.cross_attention_alpha)

    def build_attention_mask(self, attn_window):
        mask = torch.ones(
            self.visual_length,
            self.visual_length,
            dtype=torch.bool
        )
        for start in range(0, self.visual_length, attn_window):
            end = min(start + attn_window, self.visual_length)
            mask[start:end, start:end] = False

        return mask

    def adj4(self, x, seq_len):
        soft = nn.Softmax(1)
        x2 = x.matmul(x.permute(0, 2, 1)) # B*T*T
        x_norm = torch.norm(x, p=2, dim=2, keepdim=True)  # B*T*1
        x_norm_x = x_norm.matmul(x_norm.permute(0, 2, 1))
        x2 = x2/(x_norm_x+1e-20)
        output = torch.zeros_like(x2)
        if seq_len is None:
            for i in range(x.shape[0]):
                tmp = x2[i]
                adj2 = tmp
                adj2 = F.threshold(adj2, 0.7, 0)
                adj2 = soft(adj2)
                output[i] = adj2
        else:
            for i in range(len(seq_len)):
                tmp = x2[i, :seq_len[i], :seq_len[i]]
                adj2 = tmp
                adj2 = F.threshold(adj2, 0.7, 0)
                adj2 = soft(adj2)
                output[i, :seq_len[i], :seq_len[i]] = adj2

        return output

    def encode_video(self, images, padding_mask, lengths):
        images = images.to(torch.float)
        padding_mask = padding_mask.to(
            dtype=torch.bool,
            device=images.device
        ) if padding_mask is not None else build_padding_mask(
            lengths,
            images.shape[1],
            images.device
        )
        position_ids = torch.arange(images.shape[1], device=images.device)
        position_ids = position_ids.unsqueeze(0).expand(images.shape[0], -1)
        frame_position_embeddings = self.frame_position_embeddings(position_ids)
        frame_position_embeddings = frame_position_embeddings.permute(1, 0, 2)
        images = images.permute(1, 0, 2) + frame_position_embeddings

        x, _ = self.temporal((images, padding_mask))
        x = x.permute(1, 0, 2)
        valid_mask = ~padding_mask
        x = x.masked_fill(~valid_mask.unsqueeze(-1), 0)
        residual = x

        adj = self.adj4(x, lengths)
        disadj = self.disAdj(x.shape[0], x.shape[1])
        pair_mask = valid_mask.unsqueeze(2) & valid_mask.unsqueeze(1)
        adj = adj.masked_fill(~pair_mask, 0)
        disadj = disadj.masked_fill(~pair_mask, 0)

        x1_h = self.gelu(self.gc1(x, adj))
        x2_h = self.gelu(self.gc3(x, disadj))

        x1 = self.gelu(self.gc2(x1_h, adj))
        x2 = self.gelu(self.gc4(x2_h, disadj))

        x = torch.cat((x1, x2), 2)
        x = self.linear(x)
        x = x + residual
        x = x.masked_fill(~valid_mask.unsqueeze(-1), 0)

        return x

    def encode_textprompt(self, text):
        word_tokens = clip.tokenize(text).to(self.device)
        word_embeddings = self.clipmodel.encode_token(word_tokens)
        return self.clipmodel.encode_text(word_embeddings, word_tokens)

    def encode_prompt_pairs(self, positive_text, negative_text, visual_features, padding_mask, lengths):
        if len(positive_text) != len(negative_text):
            raise ValueError('Positive and negative prompt counts must match')

        positive_features = self.encode_textprompt(positive_text)
        negative_features = self.encode_textprompt(negative_text)
        text_dtype = visual_features.dtype

        visual_aligned = F.normalize(self.visual_alignment(visual_features), dim=-1)
        positive_aligned = F.normalize(self.text_alignment(positive_features.to(text_dtype)), dim=-1)
        negative_aligned = F.normalize(self.text_alignment(negative_features.to(text_dtype)), dim=-1)

        if padding_mask is None:
            padding_mask = build_padding_mask(
                lengths,
                visual_features.shape[1],
                visual_features.device
            )
        else:
            padding_mask = padding_mask.to(
                dtype=torch.bool,
                device=visual_features.device
            )
        empty_sequences = padding_mask.all(dim=1)
        padding_mask[empty_sequences, 0] = False

        pair_queries = torch.cat([positive_aligned, negative_aligned], dim=0)
        pair_queries = pair_queries.unsqueeze(1).expand(-1, visual_features.shape[0], -1)
        video_tokens = visual_aligned.permute(1, 0, 2)
        attended_pairs, _ = self.pair_cross_attention(
            pair_queries,
            video_tokens,
            video_tokens,
            key_padding_mask=padding_mask,
            need_weights=False
        )
        attended_pairs = F.normalize(
            pair_queries + torch.sigmoid(self.cross_attention_alpha) * attended_pairs,
            dim=-1
        ).permute(1, 0, 2)
        positive_attended, negative_attended = attended_pairs.chunk(2, dim=1)

        positive_similarity = torch.einsum('btd,bcd->btc', visual_aligned, positive_attended)
        negative_similarity = torch.einsum('btd,bcd->btc', visual_aligned, negative_attended)
        pair_logits = (positive_similarity - negative_similarity) / 0.07

        return positive_features, negative_features, pair_logits

    def forward(self, visual, padding_mask, text, lengths, negative_text=None):
        visual_features = self.encode_video(visual, padding_mask, lengths)
        logits1 = self.classifier(visual_features + self.mlp2(visual_features))

        if negative_text is not None:
            if not self.use_prompt_pairs:
                raise ValueError('Prompt-pair mode was not enabled when constructing VPTeAVAD')
            positive_features, negative_features, pair_logits = self.encode_prompt_pairs(
                text, negative_text, visual_features, padding_mask, lengths
            )
            return positive_features, negative_features, logits1, pair_logits

        text_features_ori = self.encode_textprompt(text)

        text_features = text_features_ori
        logits_attn = logits1.permute(0, 2, 1)
        visual_attn = logits_attn @ visual_features
        visual_attn = visual_attn / visual_attn.norm(dim=-1, keepdim=True)
        visual_attn = visual_attn.expand(visual_attn.shape[0], text_features_ori.shape[0], visual_attn.shape[2])
        text_features = text_features_ori.unsqueeze(0)
        text_features = text_features.expand(visual_attn.shape[0], text_features.shape[1], text_features.shape[2])
        text_features = text_features + visual_attn
        text_features = text_features + self.mlp1(text_features)

        visual_features_norm = visual_features / visual_features.norm(dim=-1, keepdim=True)
        text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features_norm = text_features_norm.permute(0, 2, 1)
        logits2 = visual_features_norm @ text_features_norm.type(visual_features_norm.dtype) / 0.07

        return text_features_ori, logits1, logits2
    
