#!/usr/bin/env python

# Copyright 2025 HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
SmolVLA:

[Paper](https://huggingface.co/papers/2506.01844)

Designed by Hugging Face.

Install smolvla extra dependencies:
```bash
pip install -e ".[smolvla]"
```

Example of finetuning the smolvla pretrained model (`smolvla_base`):
```bash
lerobot-train \
--policy.path=lerobot/smolvla_base \
--dataset.repo_id=<USER>/svla_so100_task1_v3 \
--batch_size=64 \
--steps=200000
```

Example of finetuning a smolVLA. SmolVLA is composed of a pretrained VLM,
and an action expert.
```bash
lerobot-train \
--policy.type=smolvla \
--dataset.repo_id=<USER>/svla_so100_task1_v3 \
--batch_size=64 \
--steps=200000
```

Example of using the smolvla pretrained model outside LeRobot training framework:
```python
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
```

"""

import math
from collections import deque
from typing import TypedDict, Unpack

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.rtc.modeling_rtc import RTCProcessor
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.smolvlm_with_expert import SmolVLMWithExpertModel
from lerobot.policies.utils import (
    populate_queues,
)
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
from lerobot.utils.device_utils import get_safe_dtype
from lerobot.policies.smolvla.attn_weight_generator import AttentionWeightGenerator, build_mlp

class ActionSelectKwargs(TypedDict, total=False):
    inference_delay: int | None
    prev_chunk_left_over: Tensor | None
    execution_horizon: int | None


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks
    return att_2d_masks


def resize_with_pad(img, width, height, pad_value=-1):
    # assume no-op when width height fits already
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but {img.shape}")

    cur_height, cur_width = img.shape[2:]

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_img = F.interpolate(
        img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
    )

    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))

    # pad on left and top of image
    padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
    return padded_img


def pad_vector(vector, new_dim):
    """Can be (batch_size x sequence_length x features_dimension)
    or (batch_size x features_dimension)
    """
    if vector.shape[-1] == new_dim:
        return vector
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    new_vector = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)
    new_vector[..., :current_dim] = vector
    return new_vector


def normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def safe_arcsin(value):
    # This ensures that the input stays within
    # [−1,1] to avoid invalid values for arcsin
    return torch.arcsin(torch.clamp(value, -1.0, 1.0))


def aloha_gripper_to_angular(value):
    # Aloha transforms the gripper positions into a linear space. The following code
    # reverses this transformation to be consistent with smolvla which is pretrained in
    # angular space.
    #
    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSED
    value = unnormalize(value, min_val=0.01844, max_val=0.05800)

    # This is the inverse of the angular to linear transformation inside the Interbotix code.
    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return safe_arcsin(value)

    # The constants are taken from the Interbotix code.
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)

    # Normalize to [0, 1].
    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    return normalize(value, min_val=0.4, max_val=1.5)


def aloha_gripper_from_angular(value):
    # Convert from the gripper position used by smolvla to the gripper position that is used by Aloha.
    # Note that the units are still angular but the range is different.

    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    value = unnormalize(value, min_val=0.4, max_val=1.5)

    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
    return normalize(value, min_val=-0.6213, max_val=1.4910)


def aloha_gripper_from_angular_inv(value):
    # Directly inverts the gripper_from_angular function.
    value = unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return normalize(value, min_val=0.4, max_val=1.5)


class SmolVLAPolicy(PreTrainedPolicy):
    """Wrapper class around VLAFlowMatching model to train and run inference within LeRobot."""

    config_class = SmolVLAConfig
    name = "smolvla"

    def __init__(
        self,
        config: SmolVLAConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,  # <--- 必須顯式接收 dataset_stats
        **kwargs,                                                    # <--- 接收其他多餘的關鍵字參數
    ):
        """
        Args:
            config: Policy configuration class instance.
            dataset_stats: Dataset statistics for normalization.
        """      
        
        super().__init__(config, dataset_stats=dataset_stats, **kwargs)
        config.validate_features()
        self.config = config
        self.init_rtc_processor()
        self.model = VLAFlowMatching(config, rtc_processor=self.rtc_processor)
        self.reset()
        self.model.to(dtype=torch.bfloat16)

    def reset(self):
        """環境 reset 時將跨步循環狀態初始化為 None"""
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }
        self._recurrent_state = None

    def init_rtc_processor(self):
        """Initialize RTC processor if RTC is enabled in config."""
        self.rtc_processor = None

        # Lets create processor if the config provided
        # If RTC is not enabled - we still can track the denoising data
        if self.config.rtc_config is not None:
            self.rtc_processor = RTCProcessor(self.config.rtc_config)

            # In case of calling init_rtc_processor after the model is created
            # We need to set the rtc_processor to the model
            # During the normal initialization process the model is not created yet
            model_value = getattr(self, "model", None)
            if model_value is not None:
                model_value.rtc_processor = self.rtc_processor

    def get_optim_params(self) -> dict:
        return self.parameters()

    @torch.no_grad()
    def _get_action_chunk(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        for k in batch:
            if k in self._queues and k != ACTION:
                batch[k] = torch.stack(list(self._queues[k]), dim=1)

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        # 推論時將內部維持的 _recurrent_state 送入，並更新當前循環狀態
        actions, next_r = self.model.sample_actions_with_recurrent(
            images, img_masks, lang_tokens, lang_masks, state,
            recurrent_state=self._recurrent_state, noise=noise, **kwargs
        )
        self._recurrent_state = next_r

        original_action_dim = self.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]
        if self.config.adapt_to_pi_aloha:
            actions = self._pi_aloha_encode_actions(actions)

        return actions

    def _prepare_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])

        return batch

    @torch.no_grad()
    def predict_action_chunk(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        self.eval()

        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        actions = self._get_action_chunk(batch, noise, **kwargs)
        return actions

    @torch.no_grad()
    def select_action(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """

        assert not self._rtc_enabled(), (
            "RTC is not supported for select_action, use it with predict_action_chunk"
        )

        self.eval()
        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        if self._check_get_actions_condition():
            actions = self._get_action_chunk(batch, noise)

            # `self.predict_action_chunk` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])

        return self._queues[ACTION].popleft()

    def _check_get_actions_condition(self) -> bool:
        return len(self._queues[ACTION]) == 0

    def _rtc_enabled(self) -> bool:
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def forward(
        self, batch: dict[str, Tensor], noise=None, time=None, reduction: str = "mean"
    ) -> dict[str, Tensor]:
        """
        支援 POMDP 軌跡時序展開 (T=4) 與截斷時間反向傳播 (TBPTT)
        """
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])

        # 時序步數由配置決定 (預設 4 步)，不再受限於 use_ava
        T = getattr(self.config, "ava_tbptt_steps", 4)
        detach_step = getattr(self.config, "ava_detach_step", 2)
        total_flow_loss = 0.0
        total_reg_loss = 0.0
        r_prev = None

        for t in range(T):
            # TBPTT 截斷歷史計算圖 (始終執行)
            if t == detach_step and r_prev is not None:
                r_prev = r_prev.detach()

            images, img_masks = self.prepare_images(batch)
            state = self.prepare_state(batch)
            lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
            lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
            actions = self.prepare_action(batch)

            flow_losses, omega_t, r_next = self.model.forward(
                images, img_masks, lang_tokens, lang_masks, state, actions,
                noise=noise, time=time, recurrent_state=r_prev
            )
            r_prev = r_next

            # 計算 Flow Matching Loss
            original_action_dim = self.config.action_feature.shape[0]
            flow_losses = flow_losses[:, :, :original_action_dim]
            actions_is_pad = batch.get("action_is_pad")
            if actions_is_pad is not None:
                flow_losses = flow_losses * (~actions_is_pad).unsqueeze(-1)
            flow_losses = flow_losses[:, :, : self.config.max_action_dim]
            step_flow_loss = flow_losses.mean()
            total_flow_loss = total_flow_loss + step_flow_loss

            # AVA 正則化損失 (僅在啟用 AVA 且有 omega_t 時計算)
            if getattr(self.config, "use_ava", False) and omega_t is not None:
                c = self.config.ava_reg_target_c
                l_omega = torch.norm(omega_t.mean() - c, p=2)
                total_reg_loss = total_reg_loss + l_omega

        # 總損失: 沒開 AVA 時只計算 Flow Loss
        lambda_reg = self.config.ava_lambda_reg if getattr(self.config, "use_ava", False) else 0.0
        total_loss = (total_flow_loss / T) + lambda_reg * (total_reg_loss / T)

        loss_dict = {
            "loss": total_loss.item(),
            "flow_loss": (total_flow_loss / T).item(),
            "reg_loss": (total_reg_loss / T).item() if getattr(self.config, "use_ava", False) else 0.0,
        }
        return total_loss, loss_dict

    def prepare_images(self, batch):
        """Apply SmolVLA preprocessing to the images, like resizing to 224x224 and padding to keep aspect ratio, and
        convert pixel range from [0.0, 1.0] to [-1.0, 1.0] as requested by SigLIP.
        """
        images = []
        img_masks = []
        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. (batch: {batch.keys()}) (image_features:{self.config.image_features})"
            )
        # Preprocess image features present in the batch
        for key in present_img_keys:
            img = batch[key][:, -1, :, :, :] if batch[key].ndim == 5 else batch[key]
            if self.config.resize_imgs_with_padding is not None:
                img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)

            # Normalize from range [0,1] to [-1,1] as expacted by siglip
            img = img * 2.0 - 1.0

            bsize = img.shape[0]
            device = img.device
            if f"{key}_padding_mask" in batch:
                mask = batch[f"{key}_padding_mask"].bool()
            else:
                mask = torch.ones(bsize, dtype=torch.bool, device=device)
            images.append(img)
            img_masks.append(mask)

        # Create image features not present in the batch
        # as fully 0 padded images.
        for num_empty_cameras in range(len(missing_img_keys)):
            if num_empty_cameras >= self.config.empty_cameras:
                break
            img = torch.ones_like(img) * -1
            mask = torch.zeros_like(mask)
            images.append(img)
            img_masks.append(mask)
        return images, img_masks

    def _pi_aloha_decode_state(self, state):
        # Flip the joints.
        for motor_idx in [1, 2, 8, 9]:
            state[:, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            state[:, motor_idx] = aloha_gripper_to_angular(state[:, motor_idx])
        return state

    def _pi_aloha_encode_actions(self, actions):
        # Flip the joints.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular(actions[:, :, motor_idx])
        return actions

    def _pi_aloha_encode_actions_inv(self, actions):
        # Flip the joints again.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular_inv(actions[:, :, motor_idx])
        return actions

    def prepare_state(self, batch):
        """Pad state"""
        state = batch[OBS_STATE][:, -1, :] if batch[OBS_STATE].ndim > 2 else batch[OBS_STATE]
        state = pad_vector(state, self.config.max_state_dim)
        return state

    def prepare_action(self, batch):
        """Pad action"""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions

    def _get_default_peft_targets(self) -> dict[str, any]:
        """
        對齊 AVA 原論文第二階段設定:
        1. 針對 LLM backbone、Vision encoder、Action head、Proprioceptive projector 套用 rank=32 LoRA
        2. AVA 專屬模組 (recurrent_projector_B, ava_generator, vlm_act_to_expert_proj)
           維持全參數最佳化 (放入 modules_to_save)
        """
        # 1. 匹配 VLM (視覺與語言層) + Action Expert 的注意力與前饋投影矩陣
        # 2. 匹配 state_proj (本體感覺投影層)
        # 3. 匹配 action_in_proj / action_out_proj / action_time_mlp (動作頭投影)
        target_modules = (
            r".*(q_proj|k_proj|v_proj|out_proj|fc1|fc2|proj)$"
            r"|.*state_proj.*"
            r"|.*action_in_proj.*"
            r"|.*action_out_proj.*"
            r"|.*action_time_mlp_in.*"
            r"|.*action_time_mlp_out.*"
        )

        # 論文規定: while fully optimizing the proposed AVA mechanism
        # 這些模組不套用 LoRA，而是進行完整的全參數更新並儲存在 checkpoint 中
        modules_to_save = [
            "model.recurrent_projector_B",
            "model.ava_generator",
            "model.vlm_act_to_expert_proj",
        ]

        return {
            "target_modules": target_modules,
            "modules_to_save": modules_to_save,
        }

    def _validate_peft_config(self, peft_config) -> None:
        """Validate PEFT configuration for SmolVLA."""
        super()._validate_peft_config(peft_config)
        if not self.config.load_vlm_weights:
            import logging

            logging.warning(
                "Training SmolVLA from scratch using PEFT. This is unlikely to yield good results. "
                "Set `load_vlm_weights=True` to fine-tune the existing policy."
            )


def pad_tensor(tensor, max_len, pad_value=0):
    """
    Efficiently pads a tensor along sequence dimension to match max_len.

    Args:
        tensor (torch.Tensor): Shape (B, L, ...) or (B, L).
        max_len (int): Fixed sequence length.
        pad_value (int/float): Value for padding.

    Returns:
        torch.Tensor: Shape (B, max_len, ...) or (B, max_len).
    """
    b, d = tensor.shape[:2]

    # Create a padded tensor of max_len and copy the existing values
    padded_tensor = torch.full(
        (b, max_len, *tensor.shape[2:]), pad_value, dtype=tensor.dtype, device=tensor.device
    )
    padded_tensor[:, :d] = tensor  # Efficient in-place copy

    return padded_tensor


class VLAFlowMatching(nn.Module):
    """
    SmolVLA

    [Paper]()

    Designed by Hugging Face.
    ┌──────────────────────────────┐
    │                 actions      │
    │                    ▲         │
    │ ┌─────────┐      ┌─|────┐    │
    │ |         │────► │      │    │
    │ |         │ kv   │      │    │
    │ |         │────► │Action│    │
    │ |   VLM   │cache │Expert│    |
    │ │         │────► |      │    │
    │ │         │      │      │    │
    │ └▲──▲───▲─┘      └───▲──┘    |
    │  │  |   |            │       |
    │  |  |   |          noise     │
    │  │  │ state                  │
    │  │ language tokens           │
    │  image(s)                    │
    └──────────────────────────────┘
    """

    def __init__(
        self,
        config: SmolVLAConfig,
        rtc_processor: RTCProcessor | None = None,
    ):
        super().__init__()
        self.config = config
        self.rtc_processor = rtc_processor  # ✅ 保存在實例變數即可
        vlm_dim = 2048 # SmolVLM2 text hidden_size

        self.vlm_with_expert = SmolVLMWithExpertModel(
            model_id=self.config.vlm_model_name,
            freeze_vision_encoder=self.config.freeze_vision_encoder,
            train_expert_only=self.config.train_expert_only,
            load_vlm_weights=self.config.load_vlm_weights,
            attention_mode=self.config.attention_mode,
            num_expert_layers=self.config.num_expert_layers,
            num_vlm_layers=self.config.num_vlm_layers,
            self_attn_every_n_layers=self.config.self_attn_every_n_layers,
            expert_width_multiplier=self.config.expert_width_multiplier,
            device=self.config.device if self.config.device is not None else "auto",
        )
        self.state_proj = nn.Linear(
            self.config.max_state_dim, self.vlm_with_expert.config.text_config.hidden_size
        )
        self.action_in_proj = nn.Linear(self.config.max_action_dim, self.vlm_with_expert.expert_hidden_size)
        self.action_out_proj = nn.Linear(self.vlm_with_expert.expert_hidden_size, self.config.max_action_dim)

        self.action_time_mlp_in = nn.Linear(
            self.vlm_with_expert.expert_hidden_size * 2, self.vlm_with_expert.expert_hidden_size
        )
        self.action_time_mlp_out = nn.Linear(
            self.vlm_with_expert.expert_hidden_size, self.vlm_with_expert.expert_hidden_size
        )

        # ==================== 1. 遞迴狀態投影器 B (始終初始化，維持時序基底) ====================
        vlm_dim = self.vlm_with_expert.config.text_config.hidden_size # 960
        self.recurrent_projector_B = build_mlp(
            vlm_dim, vlm_dim, hidden_size=vlm_dim, depth=2, out_act=False
        )

        # ==================== 2. AVA 專屬模組 (僅在啟用 AVA 時初始化) ====================
        self.ava_generator = None
        self.vlm_act_to_expert_proj = None

        if getattr(self.config, "use_ava", False):
            # (A) 映射動作生成輸入: 將 VLM 動作隱狀態投影至 Action Expert 維度
            self.vlm_act_to_expert_proj = nn.Linear(vlm_dim, self.vlm_with_expert.expert_hidden_size)

            # (B) 動態探測視覺特徵維度 (SigLIP 預設為 960)
            vision_dim = 960
            try:
                device = next(self.vlm_with_expert.parameters()).device
                dtype = next(self.vlm_with_expert.parameters()).dtype
                dummy_img = torch.zeros(1, 3, *self.config.resize_imgs_with_padding, device=device, dtype=dtype)
                vision_dim = self.vlm_with_expert.embed_image(dummy_img).shape[-1]
            except Exception:
                if hasattr(self.vlm_with_expert.config, "vision_config"):
                    vision_dim = getattr(self.vlm_with_expert.config.vision_config, "hidden_size", 960)

            # (C) VLM 軟權重生成器 (計算 omega_t)
            self.ava_generator = AttentionWeightGenerator(
                embed_dim=self.config.ava_hidden_dim, 
                image_shape=(8, 8, vision_dim),
                action_shape=(self.config.ava_chunk_len, self.config.ava_action_dim, vlm_dim), 
                text_dim=vlm_dim,
                num_images=len(self.config.image_features),
                score_config=self.config.ava_score_config,
            )
        # ==================================================================================

        self.set_requires_grad()
        self.fake_image_token = self.vlm_with_expert.processor.tokenizer.fake_image_token_id
        self.global_image_token = self.vlm_with_expert.processor.tokenizer.global_image_token_id
        self.global_image_start_token = torch.tensor(
            [self.fake_image_token, self.global_image_token], dtype=torch.long
        )

        self.add_image_special_tokens = self.config.add_image_special_tokens
        self.image_end_token = torch.tensor([self.fake_image_token], dtype=torch.long)
        self.prefix_length = self.config.prefix_length
        self.rtc_processor = rtc_processor

    def _rtc_enabled(self):
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def set_requires_grad(self):
        for params in self.state_proj.parameters():
            params.requires_grad = self.config.train_state_proj

    def sample_noise(self, shape, device):
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )
        return noise

    def sample_time(self, bsize, device):
        beta_dist = torch.distributions.Beta(concentration1=1.5, concentration0=1.0)
        time_beta = beta_dist.sample((bsize,)).to(device=device, dtype=torch.float32)
        time = time_beta * 0.999 + 0.001
        return time

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks, state: torch.Tensor = None, recurrent_state: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        embs = []
        pad_masks = []
        att_masks = []
        num_visual_tokens = 0
        all_img_embs = []

        for _img_idx, (img, img_mask) in enumerate(zip(images, img_masks, strict=False)):
            if self.add_image_special_tokens:
                image_start_token = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.global_image_start_token.to(device=self.vlm_with_expert.vlm.device)
                    )
                    .unsqueeze(0).expand(img.shape[0], -1, -1)
                )
                image_start_mask = torch.ones_like(image_start_token[:, :, 0], dtype=torch.bool)
                att_masks += [0] * (image_start_mask.shape[-1])
                embs.append(image_start_token)
                pad_masks.append(image_start_mask)

            img_emb = self.vlm_with_expert.embed_image(img)
            img_emb_dim = img_emb.shape[-1]
            img_emb = img_emb * torch.tensor(img_emb_dim**0.5, dtype=img_emb.dtype, device=img_emb.device)

            bsize, num_img_embs = img_emb.shape[:2]
            num_visual_tokens += num_img_embs
            all_img_embs.append(img_emb)

            img_mask = img_mask[:, None].expand(bsize, num_img_embs)
            embs.append(img_emb)
            pad_masks.append(img_mask)
            att_masks += [0] * num_img_embs

            if self.add_image_special_tokens:
                image_end_token = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.image_end_token.to(device=self.vlm_with_expert.vlm.device)
                    )
                    .unsqueeze(0).expand(img.shape[0], -1, -1)
                )
                image_end_mask = torch.ones_like(image_end_token[:, :, 0], dtype=torch.bool)
                embs.append(image_end_token)
                pad_masks.append(image_end_mask)
                att_masks += [0] * (image_end_mask.shape[1])

        lang_emb = self.vlm_with_expert.embed_language_tokens(lang_tokens)
        lang_emb = lang_emb * math.sqrt(lang_emb.shape[-1])
        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        att_masks += [0] * lang_emb.shape[1]

        state_emb = self.state_proj(state)
        state_emb = state_emb[:, None, :] if state_emb.ndim == 2 else state_emb
        embs.append(state_emb)
        bsize = state_emb.shape[0]
        states_seq_len = state_emb.shape[1]
        pad_masks.append(torch.ones(bsize, states_seq_len, dtype=torch.bool, device=state.device))
        att_masks += [1] * states_seq_len

        # ==================== 拼接 70 個動作佔位符 Tokens (始終常駐) ====================
        la = getattr(self.config, "ava_action_tokens_len", 70)
        d = self.vlm_with_expert.config.text_config.hidden_size
        if recurrent_state is None:
            # t = 0 步: p_0 = 0
            placeholders = torch.zeros((bsize, la, d), dtype=state_emb.dtype, device=state.device)
        else:
            # t > 0 步: p_t = r_{t-1}
            placeholders = recurrent_state.to(dtype=state_emb.dtype, device=state.device)

        embs.append(placeholders)
        pad_masks.append(torch.ones(bsize, la, dtype=torch.bool, device=state.device))
        att_masks += [1] * la
        # =============================================================================

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)[None, :].expand(bsize, -1)

        token_slices = {
            "num_visual_tokens": num_visual_tokens,
            "visual_embs": torch.cat(all_img_embs, dim=1),
            "lang_embs": lang_emb,
        }
        return embs, pad_masks, att_masks, token_slices

    def build_ava_attn_bias(self, omega_t, total_len, num_visual_tokens):
        """
        將視覺軟權重 omega_t 轉為對數加性注意力偏置矩陣
        """
        B = omega_t.shape[0]
        # 初始化全零矩陣
        attn_bias = torch.zeros((B, 1, total_len, total_len), dtype=torch.float32, device=omega_t.device)
        # omega_t: [B, L_I, 1] -> log(omega_t + 1e-8)
        log_omega = torch.log(omega_t.squeeze(-1).clamp(min=1e-8)) # [B, L_I]
        # 在視覺 Key/Value 位置加上 log(omega)
        attn_bias[:, :, :, :num_visual_tokens] = log_omega[:, None, None, :]
        return attn_bias

    def sample_actions_with_recurrent(
        self, images, img_masks, lang_tokens, lang_masks, state,
        recurrent_state=None, noise=None, **kwargs
    ):
        bsize = state.shape[0]
        device = state.device

        if noise is None:
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks, slices = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state, recurrent_state=recurrent_state
        )

        # 1. AVA 軟權重注意力偏置 (僅在啟用 AVA 時計算)
        attn_bias = None
        if getattr(self.config, "use_ava", False) and self.ava_generator is not None:
            _, (omega_t, _) = self.ava_generator(
                image_tokens=slices["visual_embs"],
                action_tokens=recurrent_state,
                text_tokens=slices["lang_embs"],
            )
            total_len = prefix_embs.shape[1] + self.config.chunk_size
            attn_bias = self.build_ava_attn_bias(omega_t, total_len, slices["num_visual_tokens"])

        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        _, past_key_values, extracted_vlm_hidden = self.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
            attn_bias=attn_bias,
            extract_layer_idx=self.config.ava_layer_idx,
        )

        # 2. 擷取並更新下一個循環狀態 r_t (始終常駐)
        la = getattr(self.config, "ava_action_tokens_len", 70)
        h_t_act = extracted_vlm_hidden[:, -la:, :]
        proj_b_dtype = next(self.recurrent_projector_B.parameters()).dtype
        next_recurrent_state = self.recurrent_projector_B(h_t_act.to(dtype=proj_b_dtype))

        # Flow matching 解碼迴圈
        num_steps = self.config.num_steps
        dt = -1.0 / num_steps
        x_t = noise
        for step in range(num_steps):
            time = 1.0 + step * dt
            time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)
            v_t = self.denoise_step(
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                x_t=x_t,
                timestep=time_tensor,
            )
            x_t = x_t + dt * v_t

        return x_t, next_recurrent_state

    def embed_suffix(self, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        # Fuse timestep + action information using an MLP
        action_emb = self.action_in_proj(noisy_actions)
        device = action_emb.device
        bsize = action_emb.shape[0]
        dtype = action_emb.dtype
        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.vlm_with_expert.expert_hidden_size,
            self.config.min_period,
            self.config.max_period,
            device=device,
        )
        time_emb = time_emb.type(dtype=dtype)

        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)

        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)  # swish == silu
        action_time_emb = self.action_time_mlp_out(action_time_emb)

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] * self.config.chunk_size
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))
        return embs, pad_masks, att_masks

    def forward(
        self, images, img_masks, lang_tokens, lang_masks, state, actions,
        noise=None, time=None, recurrent_state=None
    ):
        """單步 forward：計算 Flow Matching Loss、抽取 h^t_act 生成 r_t，以及 AVA 權重"""
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks, slices = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state, recurrent_state=recurrent_state
        )

        # ---------------- 1. AVA 專屬: VLM 軟權重注意力偏置計算 ----------------
        omega_t = None
        attn_bias = None
        if getattr(self.config, "use_ava", False) and self.ava_generator is not None:
            _, (omega_t, _) = self.ava_generator(
                image_tokens=slices["visual_embs"],
                action_tokens=recurrent_state,
                text_tokens=slices["lang_embs"],
            )
            total_len = prefix_embs.shape[1] + self.config.chunk_size
            attn_bias = self.build_ava_attn_bias(omega_t, total_len, slices["num_visual_tokens"])

        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, time)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        outputs_embeds, _, extracted_vlm_hidden = self.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
            attn_bias=attn_bias,
            extract_layer_idx=self.config.ava_layer_idx,
        )

        # ---------------- 2. 時序基底: 提取動作隱狀態並更新遞迴狀態 r_t (始終常駐) ----------------
        la = getattr(self.config, "ava_action_tokens_len", 70)
        h_t_act = extracted_vlm_hidden[:, -la:, :] # [B, 70, vlm_dim]
        proj_b_dtype = next(self.recurrent_projector_B.parameters()).dtype
        next_recurrent_state = self.recurrent_projector_B(h_t_act.to(dtype=proj_b_dtype)) # r_t = B(h^t_act)

        # ---------------- 3. AVA 專屬: 映射動作生成輸入 (Action Expert 雜訊引導) ----------------
        suffix_out = outputs_embeds[1][:, -self.config.chunk_size :]

        if getattr(self.config, "use_ava", False) and self.vlm_act_to_expert_proj is not None:
            proj_expert_dtype = next(self.vlm_act_to_expert_proj.parameters()).dtype
            act_expert_guidance = self.vlm_act_to_expert_proj(h_t_act.to(dtype=proj_expert_dtype))
            act_expert_guidance = F.adaptive_avg_pool1d(
                act_expert_guidance.transpose(1, 2), self.config.chunk_size
            ).transpose(1, 2)
            suffix_out = suffix_out + act_expert_guidance

        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        flow_losses = F.mse_loss(u_t, v_t, reduction="none")

        return flow_losses, omega_t, next_recurrent_state

    def sample_actions(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise=None,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = state.shape[0]
        device = state.device

        if noise is None:
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        # Compute image and language key value cache
        _, past_key_values = self.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )
        num_steps = self.config.num_steps
        dt = -1.0 / num_steps

        x_t = noise
        for step in range(num_steps):
            time = 1.0 + step * dt
            time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)

            def denoise_step_partial_call(input_x_t, current_timestep=time_tensor):
                return self.denoise_step(
                    x_t=input_x_t,
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                    timestep=current_timestep,
                )

            if self._rtc_enabled():
                inference_delay = kwargs.get("inference_delay")
                prev_chunk_left_over = kwargs.get("prev_chunk_left_over")
                execution_horizon = kwargs.get("execution_horizon")

                v_t = self.rtc_processor.denoise_step(
                    x_t=x_t,
                    prev_chunk_left_over=prev_chunk_left_over,
                    inference_delay=inference_delay,
                    time=time,
                    original_denoise_step_partial=denoise_step_partial_call,
                    execution_horizon=execution_horizon,
                )
            else:
                v_t = denoise_step_partial_call(x_t)

            x_t = x_t + dt * v_t

            if self.rtc_processor is not None and self.rtc_processor.is_debug_enabled():
                self.rtc_processor.track(time=time, x_t=x_t, v_t=v_t)

        return x_t

    def denoise_step(
        self,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        outputs_embeds, _, _ = self.vlm_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=self.config.use_cache,
            fill_kv_cache=False,
        )
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        return v_t
