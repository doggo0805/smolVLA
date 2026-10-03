import ast
from functools import partial
from typing import ClassVar

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

DEFAULT_NORM_LAYER = partial(nn.LayerNorm, eps=1e-6)


def build_mlp(
    in_size,
    out_size,
    hidden_size=None,
    depth=2,
    dropout=None,
    out_act=False,
    out_drop=False,
):
    if hidden_size is None:
        hidden_size = out_size

    modules = [nn.Linear(in_size, hidden_size)]
    for _ in range(2, depth):
        modules.append(nn.SiLU())
        modules.append(nn.Dropout(dropout) if dropout is not None else nn.Identity())
        modules.append(nn.Linear(hidden_size, hidden_size))

    modules += [
        nn.SiLU(),
        nn.Dropout(dropout) if dropout is not None else nn.Identity(),
        nn.Linear(hidden_size, out_size),
        nn.SiLU() if out_act else nn.Identity(),
        nn.Dropout(dropout) if out_drop and dropout is not None else nn.Identity(),
    ]
    return nn.Sequential(*modules)


def get_1d_position_embedding(size, embed_dim):
    def get_angles(pos, i, embed_dim):
        return pos / np.power(10000, (2 * (i // 2)) / embed_dim)

    angle_rads = get_angles(np.arange(size)[:, None], np.arange(embed_dim)[None, :], embed_dim)

    pos_embed = np.zeros_like(angle_rads)
    pos_embed[:, 0::2] = np.sin(angle_rads[:, 0::2])
    pos_embed[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return pos_embed


def get_2d_position_embedding(h, w, embed_dim):
    pos_embed_h = get_1d_position_embedding(h, embed_dim)
    pos_embed_w = get_1d_position_embedding(w, embed_dim)

    pos_embed = pos_embed_h[:, None, :] + pos_embed_w[None, :, :]
    return pos_embed


def apply_rope(tokens, cos_cache, sin_cache, position_ids):
    cos_cache = cos_cache[position_ids][None, None, ...]
    sin_cache = sin_cache[position_ids][None, None, ...]

    tokens_1 = tokens[..., ::2]  # (B, num_head, seq_len, dim)
    tokens_2 = tokens[..., 1::2]

    rotated_tokens_1 = tokens_1 * cos_cache[..., ::2] - tokens_2 * sin_cache[..., ::2]
    rotated_tokens_2 = tokens_1 * sin_cache[..., 1::2] + tokens_2 * cos_cache[..., 1::2]

    tokens = torch.stack([rotated_tokens_1, rotated_tokens_2], dim=-1).flatten(-2)
    return tokens


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        bias=True,
        max_position=1024,
        wo_qkv_proj=False,
        wo_out_proj=False,
        use_sdpa=True,
    ):
        super().__init__()

        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim // self.num_heads

        self.scale = self.head_dim**-0.5
        self.use_sdpa = use_sdpa

        self.wo_qkv_proj = wo_qkv_proj
        self.wo_out_proj = wo_out_proj
        if not wo_qkv_proj:
            self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
            self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
            self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
        if not wo_out_proj:
            self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
        self.apply(self._init_weights)

        self.max_position = max_position
        self._init_rope()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _init_rope(self):
        inv_freq = 1 / (10000 ** (torch.arange(0, self.head_dim, 2) / self.head_dim))
        max_seq_len = self.max_position + 128
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)

        self.register_buffer("cos_cache", emb.cos())
        self.register_buffer("sin_cache", emb.sin())

    def forward(self, query, key, value, query_position_ids=None, key_position_ids=None):
        B, N_q, _ = query.shape
        N_k = key.shape[1]

        if not self.wo_qkv_proj:
            query_states = self.q_proj(query)
            key_states = self.k_proj(key)
            value_states = self.v_proj(value)
        else:
            query_states, key_states, value_states = query, key, value

        query_states = query_states.view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(B, N_k, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(B, N_k, self.num_heads, self.head_dim).transpose(1, 2)

        # Add RoPE
        if query_position_ids is not None:
            key_position_ids = key_position_ids if key_position_ids is not None else query_position_ids
            query_states = apply_rope(query_states, self.cos_cache, self.sin_cache, query_position_ids)
            key_states = apply_rope(key_states, self.cos_cache, self.sin_cache, key_position_ids)

        if self.use_sdpa:
            attn_output = F.scaled_dot_product_attention(query_states, key_states, value_states, scale=self.scale)
            attn_weights = None
        else:
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale
            # upcast attention to fp32
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(B, N_q, self.embed_dim)

        if not self.wo_out_proj:
            attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class InjectTextModule(nn.Module):
    def __init__(self, embed_dim, norm_layer=DEFAULT_NORM_LAYER):
        super().__init__()

        self.proj = build_mlp(embed_dim, embed_dim * 2, hidden_size=embed_dim * 4, dropout=0.1)

    def forward(self, tokens, text_tokens):
        text_token = text_tokens.mean(dim=-2)
        text_token = self.proj(text_token)

        scale, shift = torch.chunk(text_token.unsqueeze_(1), 2, dim=-1)

        tokens = tokens * (1 + scale) + shift

        return tokens


# 修改 attn_weight_generator.py 中的 InjectActionModule:
class InjectActionModule(nn.Module):
    def __init__(self, embed_dim, image_shape, action_shape, num_images, norm_layer=DEFAULT_NORM_LAYER):
        super().__init__()

        self.pre_norm = norm_layer(embed_dim)
        self.pre_action_norm = norm_layer(embed_dim)
        self.cross_attn = MultiheadAttention(embed_dim, num_heads=embed_dim // 128)

        image_h, image_w, _ = image_shape
        num_patches = image_h * image_w
        action_frames, action_dim, _ = action_shape
        num_actions = action_frames * action_dim
        
        # 註冊可動態適應的 Buffer
        query_position_ids = torch.arange(num_patches * num_images)
        key_position_ids = torch.arange(num_patches * num_images, num_patches * num_images + num_actions)
        self.register_buffer("query_position_ids", query_position_ids, persistent=False)
        self.register_buffer("key_position_ids", key_position_ids, persistent=False)

    def forward(self, tokens, action_tokens):
        residual = tokens
        tokens = self.pre_norm(tokens)
        action_tokens = self.pre_action_norm(action_tokens)

        # 支援動態長度對齊
        q_len = tokens.shape[1]
        k_len = action_tokens.shape[1]
        q_pos = self.query_position_ids[:q_len] if q_len <= len(self.query_position_ids) else torch.arange(q_len, device=tokens.device)
        k_pos = self.key_position_ids[:k_len] if k_len <= len(self.key_position_ids) else torch.arange(q_len, q_len + k_len, device=tokens.device)

        attn_out, _ = self.cross_attn(
            tokens, action_tokens, action_tokens, q_pos, k_pos
        )

        tokens = residual + attn_out
        return tokens


class SelfAttentionModule(nn.Module):
    def __init__(self, embed_dim, image_shape, num_images, norm_layer=DEFAULT_NORM_LAYER):
        super().__init__()

        self.pre_norm = norm_layer(embed_dim)
        self.pre_ori_norm = norm_layer(embed_dim)
        self.self_attn = MultiheadAttention(embed_dim, num_heads=embed_dim // 128)

        image_h, image_w, _ = image_shape
        num_patches = image_h * image_w
        # Single-image input only uses the sin-cos position embedding.
        position_ids = None if num_images == 1 else torch.arange(num_patches * num_images)
        self.register_buffer("position_ids", position_ids)

    def forward(self, tokens, ori_tokens):
        residual = tokens
        tokens = self.pre_norm(tokens)

        ori_tokens = self.pre_ori_norm(ori_tokens)

        attn_out, _ = self.self_attn(tokens, ori_tokens, ori_tokens, self.position_ids)

        tokens = residual + attn_out
        return tokens


class FFN(nn.Module):
    def __init__(self, embed_dim, norm_layer=DEFAULT_NORM_LAYER):
        super().__init__()

        self.pre_norm = norm_layer(embed_dim)
        self.proj = build_mlp(embed_dim, embed_dim, hidden_size=embed_dim * 2, dropout=0.1)

    def forward(self, tokens):
        residual = tokens
        tokens = self.pre_norm(tokens)

        ffn_out = self.proj(tokens)

        tokens = residual + ffn_out
        return tokens


class AttentionWeightBlock(nn.Module):
    def __init__(self, embed_dim, image_shape, action_shape, num_images, norm_layer=DEFAULT_NORM_LAYER):
        super().__init__()

        self.inject_text = InjectTextModule(embed_dim, norm_layer=norm_layer)
        self.inject_action = InjectActionModule(embed_dim, image_shape, action_shape, num_images, norm_layer=norm_layer)
        self.self_attn = SelfAttentionModule(embed_dim, image_shape, num_images, norm_layer=norm_layer)
        self.ffn = FFN(embed_dim, norm_layer=norm_layer)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, image_tokens, action_tokens, text_tokens):
        ori_image_tokens = image_tokens

        image_tokens = self.inject_text(image_tokens, text_tokens)
        image_tokens = self.inject_action(image_tokens, action_tokens)
        image_tokens = self.self_attn(image_tokens, ori_image_tokens)

        image_tokens = self.ffn(image_tokens)
        return image_tokens


class STEHead(nn.Module):
    def __init__(self, embed_dim, max_steps=None):
        super().__init__()

        self.pre_norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, 1, bias=False)
        self._init_weights()

    def _init_weights(self):
        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0.05)

    def forward(self, tokens, selected_ratio=None, batch_idx=None):
        B, N, D = tokens.shape
        tokens = self.pre_norm(tokens.reshape(B * N, D)).reshape(B, N, D)
        logits = self.proj(tokens)  # (B, N, 1)

        if selected_ratio is None:  # Training or auto-select activated tokens during inference
            probs = torch.sigmoid(logits)
            binary_mask = (probs > 0.5).float()
            binary_mask = probs + (binary_mask - probs).detach()
            ret_value = (binary_mask, binary_mask)
        else:  # Directly select activated tokens during inference
            _, N, _ = logits.shape
            num_selected = int(N * selected_ratio)
            indexes = torch.topk(logits, k=num_selected, dim=-2)[1]
            binary_mask = torch.zeros_like(logits)
            binary_mask.scatter_(-2, indexes, 1.0)
            ret_value = binary_mask

        return ret_value


class GumbelSoftmaxHead(nn.Module):
    def __init__(self, embed_dim, init_tau=1.0, final_tau=0.1, max_steps=None):
        super().__init__()

        self.pre_norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, 2, bias=False)
        self.init_tau = init_tau
        self.final_tau = final_tau
        self.alpha = -np.log(final_tau / init_tau) / (max_steps * 0.8) if max_steps is not None else None
        self._init_weights()

    def _init_weights(self):
        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias[0], 0.05)
            nn.init.constant_(self.proj.bias[1], 0.0)

    def forward(self, tokens, selected_ratio=None, batch_idx=None):
        B, N, D = tokens.shape
        tokens = self.pre_norm(tokens.reshape(B * N, D)).reshape(B, N, D)
        logits = self.proj(tokens)  # (B, N, 2)

        if self.alpha is not None and batch_idx is not None:  # Training
            tau = float(max(self.final_tau, self.init_tau * np.exp(-self.alpha * batch_idx)))
            probs = F.gumbel_softmax(logits, tau=tau, dim=-1, hard=True)
            binary_mask = probs[:, :, [0]]

            probs = F.softmax(logits / tau, dim=-1)
            ret_value = (binary_mask, probs[:, :, [0]])
        elif selected_ratio is None:  # Automatically select activated tokens during inference
            assert not self.training
            probs = F.softmax(logits / self.final_tau, dim=-1)
            binary_mask = (probs[:, :, [0]] > probs[:, :, [1]]).float()
            ret_value = binary_mask
        else:  # Directly select activated tokens during inference
            assert not self.training
            probs = F.softmax(logits / self.final_tau, dim=-1)[:, :, [0]]
            _, N, _ = probs.shape
            num_selected = int(N * selected_ratio)
            indexes = torch.topk(probs, k=num_selected, dim=-2)[1]
            binary_mask = torch.zeros_like(probs)
            binary_mask.scatter_(-2, indexes, 1.0)
            ret_value = binary_mask

        return ret_value


def parse_score_config(score_config):
    values = ast.literal_eval(score_config) if isinstance(score_config, str) else score_config
    if not isinstance(values, (tuple, list)) or len(values) != 3:
        raise ValueError("score_config must be a tuple/list of (pos_score, neg_score, score_bias)")
    return tuple(float(value) for value in values)


class SoftmaxScoreHead(nn.Module):
    def __init__(self, embed_dim, init_tau=1.0, final_tau=0.1, max_steps=None, score_config=(1.9, 0.1, 0.0)):
        super().__init__()

        pos_score, neg_score, score_bias = parse_score_config(score_config)
        scores = (pos_score, neg_score)
        self.scores = scores
        self.score_bias = score_bias

        self.pre_norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, 2, bias=False)
        self.init_tau = init_tau
        self.final_tau = final_tau
        self.alpha = -np.log(final_tau / init_tau) / (max_steps * 0.8) if max_steps is not None else None
        self._init_weights()

    def _init_weights(self):
        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias[0], 0.05)
            nn.init.constant_(self.proj.bias[1], 0.0)

    def forward(self, tokens, selected_ratio=None, batch_idx=None):
        B, N, D = tokens.shape
        tokens = self.pre_norm(tokens.reshape(B * N, D)).reshape(B, N, D)
        logits = self.proj(tokens)  # (B, N, 2)

        if self.alpha is not None and batch_idx is not None:  # Training
            tau = float(max(self.final_tau, self.init_tau * np.exp(-self.alpha * batch_idx)))
            probs = F.softmax(logits / tau, dim=-1)
            attn_weights = self.score_bias + probs[:, :, [0]] * self.scores[0] + probs[:, :, [1]] * self.scores[1]
            ret_value = (attn_weights, probs[:, :, [0]])
        elif selected_ratio is None:  # Automatically select activated tokens during inference
            assert not self.training
            probs = F.softmax(logits / self.final_tau, dim=-1)
            attn_weights = self.score_bias + probs[:, :, [0]] * self.scores[0] + probs[:, :, [1]] * self.scores[1]
            ret_value = attn_weights
        else:  # Directly select activated tokens during inference
            assert not self.training
            probs = F.softmax(logits / self.final_tau, dim=-1)
            attn_weights = self.score_bias + probs[:, :, [0]] * self.scores[0] + probs[:, :, [1]] * self.scores[1]

            _, N, _ = attn_weights.shape
            num_pruned = int(N * (1 - selected_ratio))
            indexes = torch.topk(attn_weights, k=num_pruned, dim=-2, largest=False)[1]
            attn_weights.scatter_(-2, indexes, 0)
            ret_value = attn_weights

        return ret_value


class SigmoidScoreHead(nn.Module):
    def __init__(self, embed_dim, init_tau=1.0, final_tau=0.5, max_steps=None):
        super().__init__()

        scores = (2.0, 0.0)
        assert isinstance(scores, tuple) and len(scores) == 2
        self.scores = scores

        self.pre_norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, 1, bias=False)
        self.init_tau = init_tau
        self.final_tau = final_tau
        self.alpha = -np.log(final_tau / init_tau) / (max_steps * 0.8) if max_steps is not None else None
        self._init_weights()

    def _init_weights(self):
        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias[0], 0.05)
            nn.init.constant_(self.proj.bias[1], 0.0)

    def forward(self, tokens, selected_ratio=None, batch_idx=None):
        B, N, D = tokens.shape
        tokens = self.pre_norm(tokens.reshape(B * N, D)).reshape(B, N, D)
        logits = self.proj(tokens)  # (B, N, 1)

        if self.alpha is not None and batch_idx is not None:  # Training
            probs = torch.sigmoid(logits)
            attn_weights = self.scores[1] + (self.scores[0] - self.scores[1]) * probs
            ret_value = (attn_weights, probs)
        elif selected_ratio is None:  # Automatically select activated tokens during inference
            assert not self.training
            attn_weights = self.scores[1] + (self.scores[0] - self.scores[1]) * torch.sigmoid(logits)
            ret_value = attn_weights
        else:  # Directly select activated tokens during inference
            assert not self.training
            attn_weights = self.scores[1] + (self.scores[0] - self.scores[1]) * torch.sigmoid(logits)

            _, N, _ = attn_weights.shape
            num_pruned = int(N * (1 - selected_ratio))
            indexes = torch.topk(attn_weights, k=num_pruned, dim=-2, largest=False)[1]
            attn_weights.scatter_(-2, indexes, 0)
            ret_value = attn_weights

        return ret_value


class AttentionWeightGenerator(nn.Module):
    head_modules: ClassVar[dict[str, type[nn.Module]]] = {
        "ste": STEHead,
        "gumbel": GumbelSoftmaxHead,
        "softmaxscore": SoftmaxScoreHead,
        "sigmoidscore": SigmoidScoreHead,
    }

    def __init__(
        self,
        embed_dim,
        image_shape,
        action_shape,
        text_dim,
        num_images,
        max_steps=None,
        head_type="softmaxscore",
        score_config=(1.9, 0.1, 0.0),
        sink_ids=None,
        sink_weight=1,
    ):
        super().__init__()

        assert head_type in self.head_modules
        head_module = self.head_modules[head_type]
        head_kwargs = {"score_config": score_config} if head_type == "softmaxscore" else {}

        assert sink_ids is None or isinstance(sink_ids, (tuple, list))
        self.sink_ids = sink_ids
        self.sink_weight = sink_weight

        image_h, image_w, image_dim = image_shape
        self.image2embed = build_mlp(image_dim, embed_dim, hidden_size=embed_dim // 2)
        image_pos_embed = get_2d_position_embedding(image_h, image_w, embed_dim)
        self.register_buffer(
            "image_pos_embed",
            torch.from_numpy(image_pos_embed).reshape(1, -1, embed_dim).repeat(1, num_images, 1).float(),
            persistent=False,
        )

        self.text2embed = build_mlp(text_dim, embed_dim, hidden_size=embed_dim // 2, out_act=True)

        action_frames, action_dim, action_feature_dim = action_shape
        self.action2embed_input = build_mlp(
            action_feature_dim, action_feature_dim, hidden_size=action_feature_dim // 2, out_act=False
        )
        self.action2embed = build_mlp(action_feature_dim, embed_dim, hidden_size=embed_dim // 2, out_act=True)
        action_zeros = torch.zeros((1, action_frames * action_dim, action_feature_dim))  # (B, N, D)
        self.register_buffer("action_zeros", action_zeros, persistent=False)

        self.block = AttentionWeightBlock(embed_dim, image_shape, action_shape, num_images)
        self.head = head_module(embed_dim, max_steps=max_steps, **head_kwargs)

    def forward(self, image_tokens, action_tokens, text_tokens, selected_ratio=None, batch_idx=None):
        B, _N, _ = image_tokens.shape
        image_tokens = self.image2embed(image_tokens)
        image_tokens = image_tokens + self.image_pos_embed

        if action_tokens is None:
            input_action_tokens = self.action_zeros.repeat(B, 1, 1)
        else:
            input_action_tokens = self.action2embed_input(action_tokens)

        action_tokens = self.action2embed(input_action_tokens)

        text_tokens = self.text2embed(text_tokens)

        image_tokens = self.block(image_tokens, action_tokens, text_tokens)
        attn_weights = self.head(image_tokens, selected_ratio=selected_ratio, batch_idx=batch_idx)  # (B, N, 1)

        if not isinstance(attn_weights, tuple):
            attn_weights = (attn_weights, attn_weights.clone())

        if self.sink_ids is not None:
            target_attn_weights = attn_weights[0]

            attn_sink = target_attn_weights[:, self.sink_ids]
            mask = (attn_sink < self.sink_weight).bfloat16()
            target_attn_weights[:, self.sink_ids] = attn_sink + mask * (self.sink_weight - attn_sink.detach())

        return input_action_tokens, attn_weights