import os
import torch
import logging
from contextlib import contextmanager

# Model dep
import torch

## XUT is the custom arch used in HDM
from .xut import env

env.TORCH_COMPILE = False
env.USE_LIGER = False
env.USE_XFORMERS = False
env.USE_XFORMERS_LAYERS = False
from .xut.xut import XUDiT
from .xut.modules.axial_rope import make_axial_pos_no_cache
from transformers import Qwen3Model, Qwen2Tokenizer

# Comfy
import folder_paths
import comfy.model_management
import comfy.latent_formats as latent_formats
import comfy.sd1_clip as sd1_clip
import comfy.sd
import comfy.model_management as model_management
import comfy.model_patcher
import comfy.utils
import comfy.conds
from comfy.supported_models_base import BASE, ClipTarget
from comfy.model_base import BaseModel, ModelType


@contextmanager
def operation_patch(operations):
    """
    Directly patch torch.nn so we don't need to re-impl the model arch
    Not recommended but easiest
    """
    module_list = set(i for i in dir(torch.nn) if not i.startswith("_"))
    for op in dir(operations):
        if op in module_list:
            setattr(torch.nn, f"org_{op}", getattr(torch.nn, op))
            setattr(torch.nn, op, getattr(operations, op))
    yield
    for op in dir(operations):
        if op in module_list:
            setattr(torch.nn, op, getattr(torch.nn, f"org_{op}"))


class EQVAE(latent_formats.LatentFormat):
    """
    Latent Format for https://huggingface.co/KBlueLeaf/EQ-SDXL-VAE
    """

    def __init__(self):
        self.scale_factor = 1.0
        self.latents_mean = torch.tensor(
            [
                -3.865960634760459,
                -1.3353925720045965,
                0.8353661802812418,
                0.909888159310023,
            ]
        ).view(1, 4, 1, 1)
        self.latents_std = torch.tensor(
            [
                10.036514488374861,
                6.848430044716239,
                7.213817552213277,
                5.591315799887989,
            ]
        ).view(1, 4, 1, 1)

    def process_in(self, latent):
        latents_mean = self.latents_mean.to(latent.device, latent.dtype)
        latents_std = self.latents_std.to(latent.device, latent.dtype)
        return (latent - latents_mean) * self.scale_factor / latents_std

    def process_out(self, latent):
        latents_mean = self.latents_mean.to(latent.device, latent.dtype)
        latents_std = self.latents_std.to(latent.device, latent.dtype)
        return latent * latents_std / self.scale_factor + latents_mean


class Qwen3_600MTokenizer(sd1_clip.SDTokenizer):
    """
    Tokenizer Class for Qwen3 with TI embedding support
    """

    def __init__(self, embedding_directory=None, tokenizer_data={}):
        tokenizer_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "qwen3_tokenizer"
        )
        super().__init__(
            tokenizer_path,
            pad_with_end=False,
            embedding_size=1024,  # We need this for TI embedding
            embedding_key="qwen3",
            tokenizer_class=Qwen2Tokenizer,
            has_start_token=False,
            has_end_token=False,
            pad_to_max_length=False,
            max_length=99999999,
            min_length=1,
            pad_token=151643,
            tokenizer_data=tokenizer_data,
        )


class Qwen3_600M_Wrapper(Qwen3Model):
    """
    Qwen3 Model wrapper to support ComfyUI's API
    """

    def forward(
        self,
        input_tokens,
        attention_mask=None,
        embeds=None,
        num_tokens=None,
        intermediate_output=None,
        final_layer_norm_intermediate=True,
        dtype=None,
    ):
        # We use Autocast here to resolve dtype issue easily
        # In theory you may want to manually set dtype instead of autocast for performance
        with torch.autocast(self.device.type, dtype=self.dtype):
            result = (
                super()
                .forward(input_tokens, attention_mask, inputs_embeds=embeds)
                .last_hidden_state
            )
        # we only have last layer output, and we don't have pooled emb as well
        return result.to(dtype), None


def Qwen3_600M(config, dtype, device, operations):
    """
    Qwen3 Model Class, we init model in normal mode + operation patch
    num_layers is needed in ComfyUI's Node
    """
    with torch.inference_mode(False), operation_patch(operations):
        model = (
            Qwen3_600M_Wrapper.from_pretrained(
                "Qwen/Qwen3-0.6B", torch_dtype=dtype, attn_implementation="sdpa"
            )
            .to(device)
            .to(dtype)
        )
        model = model.to(device).eval().requires_grad_(False)
        model.num_layers = model.config.num_hidden_layers
    return model


class Qwen3_600MModel(sd1_clip.SDClipModel):
    """
    The TextEncoder class for ComfyUI
    if you have your own following API, you don't need to inheirt this class:
        process_tokens(), load_sd(), encode(), reset/set_clip_options()
    Just make a child class of sd1_clip.SD1ClipModel will be easier
    """

    def __init__(
        self,
        device="cpu",
        layer="last",
        layer_idx=None,
        dtype=None,
        attention_mask=True,
        model_options={},
    ):
        super().__init__(
            device=device,
            layer=layer,
            layer_idx=layer_idx,
            textmodel_json_config={},
            dtype=dtype,
            special_tokens={"pad": 151643},
            layer_norm_hidden_state=False,
            model_class=Qwen3_600M,
            enable_attention_masks=attention_mask,
            return_attention_masks=attention_mask,
            model_options=model_options,
        )


class HDMQwenModel(sd1_clip.SD1ClipModel):
    """
    TE(CLIP) API wrapper in ComfyUI
    """

    def __init__(self, device="cpu", dtype=None, model_options={}):
        super().__init__(
            device=device,
            dtype=dtype,
            name="qwen3_600m",  # This name will be used to find TE in state_dict
            clip_model=Qwen3_600MModel,
            model_options=model_options,
        )

    def encode_token_weights(self, token_weight_pairs):
        result = self.qwen3_600m.encode_token_weights(token_weight_pairs)
        return result


class HDMWrapper(XUDiT):
    """
    Wrapper for initialize XUDiT model and do forward in ComfyUI
    """

    def __init__(
        self,
        patch_size=2,
        input_dim=4,
        dim=1024,
        ctx_dim=1024,
        ctx_size=256,
        heads=16,
        dim_head=64,
        mlp_dim=3072,
        depth=4,
        enc_blocks=1,
        dec_blocks=3,
        dec_ctx=False,
        class_cond=0,
        shared_adaln=True,
        concat_ctx=True,
        use_dyt=False,
        double_t=False,
        addon_info_embs_dim=1,
        tread_config={
            "prev_trns_depth": 1,
            "post_trns_depth": 3,
            "dropout_ratio": 0.5,
        },
        dtype=None,
        device=None,
        operations=None,
        **kwargs,
    ):
        with torch.inference_mode(False), operation_patch(operations):
            super().__init__(
                patch_size=patch_size,
                input_dim=input_dim,
                dim=dim,
                ctx_dim=ctx_dim,
                ctx_size=ctx_size,
                heads=heads,
                dim_head=dim_head,
                mlp_dim=mlp_dim,
                depth=depth,
                enc_blocks=enc_blocks,
                dec_blocks=dec_blocks,
                dec_ctx=dec_ctx,
                class_cond=class_cond,
                shared_adaln=shared_adaln,
                concat_ctx=concat_ctx,
                use_dyt=use_dyt,
                double_t=double_t,
                addon_info_embs_dim=addon_info_embs_dim,
                tread_config=tread_config,
            )
            self.eval().requires_grad_(False)
            if device is not None:
                self.to(device)
            if dtype is not None:
                self.to(dtype)
            self.dtype = dtype
            self.device = device

    def forward(
        self,
        x,
        timesteps,
        context,
        x_shift=None,
        y_shift=None,
        zoom=None,
        *args,
        **kwargs,
    ):
        b, c, h, w = x.shape
        aspect_ratio = w / h
        aspect_ratio_info = (
            torch.tensor([aspect_ratio], dtype=x.dtype, device=x.device)
            .repeat(b, 1)
            .log()
        )

        if x_shift is not None:
            pos_map = make_axial_pos_no_cache(h, w, dtype=x.dtype, device=x.device)[
                None
            ].repeat(b, 1, 1)
            if x_shift.ndim == 1:
                x_shift = x_shift[:, None]
                y_shift = y_shift[:, None]
                zoom = zoom[:, None, None]
            pos_map[..., 1] = pos_map[..., 1] + x_shift
            pos_map[..., 0] = pos_map[..., 0] + y_shift
            pos_map = pos_map / zoom
        else:
            pos_map = None

        return super().forward(
            x.to(self.dtype),
            timesteps.to(self.dtype),
            context,
            pos_map=pos_map,
            addon_info=aspect_ratio_info,
        )


class HDMSmall(BaseModel):
    def __init__(self, model_config, model_type=ModelType.FLOW, device=None):
        super().__init__(model_config, model_type, device=device, unet_model=HDMWrapper)

    def extra_conds(self, **kwargs):
        base_result = super().extra_conds(**kwargs)
        if "x_shift" in kwargs:
            x_shift, y_shift, zoom = (
                kwargs["x_shift"],
                kwargs["y_shift"],
                kwargs["zoom"],
            )
            x_shift = torch.tensor([x_shift], dtype=torch.float32, device=self.device)
            y_shift = torch.tensor([y_shift], dtype=torch.float32, device=self.device)
            zoom = torch.tensor([zoom], dtype=torch.float32, device=self.device)
            base_result["x_shift"] = comfy.conds.CONDRegular(x_shift)
            base_result["y_shift"] = comfy.conds.CONDRegular(y_shift)
            base_result["zoom"] = comfy.conds.CONDRegular(zoom)
        return base_result


class HomeDiffusionSmall(BASE):
    unet_config = {}

    # Standard flow matching without shift
    sampling_settings = {
        "multiplier": 1.0,
        "shift": 1,
    }

    # Some random factor here, should be determined by profiling
    memory_usage_factor = 2

    unet_extra_config = {}
    latent_format = EQVAE

    # HDM is trained with fp16-amp so all dtype supported
    supported_inference_dtypes = [torch.bfloat16, torch.float16, torch.float32]

    vae_key_prefix = ["vae."]
    text_encoder_key_prefix = ["te.text_models.0."]

    def __init__(self, unet_config):
        super().__init__(unet_config)
        if comfy.model_management.extended_fp16_support():
            self.supported_inference_dtypes = [
                torch.float16
            ] + self.supported_inference_dtypes

    def get_model(self, state_dict, prefix="", device=None):
        out = HDMSmall(self, device=device)
        return out

    def clip_target(self, state_dict={}):
        return ClipTarget(Qwen3_600MTokenizer, HDMQwenModel)

    def process_clip_state_dict(self, state_dict):
        """
        Custom state dict process for loading TE
        """
        state_dict = comfy.utils.state_dict_prefix_replace(
            state_dict,
            {"te.text_models.0.": "qwen3_600m.transformer."},
            filter_keys=True,
        )
        state_dict["qwen3_600m.logit_scale"] = torch.tensor(1.0)
        return state_dict


def load_state_dict_hdm(
    sd,
    embedding_directory=None,
    model_options={},
    te_model_options={},
    metadata=None,
):
    """Directly copied from comfy.sd and simplified for HDM only"""
    clip = None
    vae = None
    model = None
    model_patcher = None
    diffusion_model_prefix = "unet.model."

    parameters = comfy.utils.calculate_parameters(sd, diffusion_model_prefix)
    weight_dtype = comfy.utils.weight_dtype(sd, diffusion_model_prefix)
    load_device = model_management.get_torch_device()

    model_config = HomeDiffusionSmall({})

    unet_weight_dtype = list(model_config.supported_inference_dtypes)
    if model_config.scaled_fp8 is not None:
        weight_dtype = None

    model_config.custom_operations = model_options.get("custom_operations", None)
    unet_dtype = model_options.get("dtype", model_options.get("weight_dtype", None))

    if unet_dtype is None:
        unet_dtype = model_management.unet_dtype(
            model_params=parameters,
            supported_dtypes=unet_weight_dtype,
            weight_dtype=weight_dtype,
        )

    manual_cast_dtype = model_management.unet_manual_cast(
        unet_dtype, load_device, model_config.supported_inference_dtypes
    )
    model_config.set_inference_dtype(unet_dtype, manual_cast_dtype)

    inital_load_device = model_management.unet_inital_load_device(
        parameters, unet_dtype
    )
    model = model_config.get_model(
        sd, diffusion_model_prefix, device=inital_load_device
    )
    model.load_model_weights(sd, diffusion_model_prefix)

    # In HDM we use SDXL vae arch so we can directly use built-in VAE class here
    # For custom VAE you will need to implement encode() and decode() method
    vae_sd = comfy.utils.state_dict_prefix_replace(
        sd, {k: "" for k in model_config.vae_key_prefix}, filter_keys=True
    )
    vae_sd = model_config.process_vae_state_dict(vae_sd)
    vae = comfy.sd.VAE(sd=vae_sd, metadata=metadata)

    clip_target = model_config.clip_target(state_dict=sd)
    if clip_target is not None:
        clip_sd = model_config.process_clip_state_dict(sd)
        parameters = comfy.utils.calculate_parameters(clip_sd)
        if parameters < 1e6:
            # no TE in ckpt, use pre-defined value:
            # 600M
            parameters = 600_000_000
        clip = comfy.sd.CLIP(
            clip_target,
            embedding_directory=embedding_directory,
            tokenizer_data=clip_sd,
            parameters=parameters,
            model_options=te_model_options,
        )
        m, u = clip.load_sd(clip_sd, full_model=True)
        if len(m) > 0:
            logging.warning(
                "HDM TE missing some weights in state dict, "
                "if you are using DiT only ckpt than this is expected."
            )

        if len(u) > 0:
            logging.debug("clip unexpected {}:".format(u))

    left_over = sd.keys()
    if len(left_over) > 0:
        logging.debug("left over keys: {}".format(left_over))

    model_patcher = comfy.model_patcher.ModelPatcher(
        model,
        load_device=load_device,
        offload_device=model_management.unet_offload_device(),
    )
    if inital_load_device != torch.device("cpu"):
        logging.info("loaded diffusion model directly to GPU")
        model_management.load_models_gpu([model_patcher], force_full_load=True)

    return (model_patcher, clip, vae)


def load_checkpoint_hdm(
    ckpt_path,
    embedding_directory=None,
    model_options={},
    te_model_options={},
):
    sd, metadata = comfy.utils.load_torch_file(ckpt_path, return_metadata=True)
    out = load_state_dict_hdm(
        sd,
        embedding_directory,
        model_options,
        te_model_options=te_model_options,
        metadata=metadata,
    )
    if out is None:
        raise RuntimeError("ERROR: Could not load HDM model")
    return out


class HDMCheckpointLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (
                    folder_paths.get_filename_list("checkpoints"),
                    {"tooltip": "The name of the checkpoint (model) to load."},
                ),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    OUTPUT_TOOLTIPS = (
        "The model used for denoising latents.",
        "The CLIP model used for encoding text prompts.",
        "The VAE model used for encoding and decoding images to and from latent space.",
    )
    FUNCTION = "load_checkpoint"

    CATEGORY = "loaders"
    DESCRIPTION = "Loads a HDM model checkpoint."

    def load_checkpoint(self, ckpt_name):
        ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
        return load_checkpoint_hdm(
            ckpt_path,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )


class HDMCameraParam:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "x_shift": (
                    "FLOAT",
                    {"default": 0.0, "min": -5.0, "max": 5.0, "step": 0.01},
                ),
                "y_shift": (
                    "FLOAT",
                    {"default": 0.0, "min": -5.0, "max": 5.0, "step": 0.01},
                ),
                "zoom": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.2, "max": 5.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    OUTPUT_TOOLTIPS = ""
    FUNCTION = "apply"

    CATEGORY = "conditioning/hdm"
    DESCRIPTION = "Apply camera parameters to conditioning"

    def apply(self, positive, negative, x_shift, y_shift, zoom):
        for pos in positive:
            pos[1]["x_shift"] = x_shift
            pos[1]["y_shift"] = y_shift
            pos[1]["zoom"] = zoom
        for neg in negative:
            neg[1]["x_shift"] = x_shift
            neg[1]["y_shift"] = y_shift
            neg[1]["zoom"] = zoom
        return positive, negative


NODE_CLASS_MAPPINGS = {
    "HDMLoader": HDMCheckpointLoader,
    "HDMCameraParam": HDMCameraParam,
}

NODE_DISPLAY_NAME_MAPPINGS = {"HDMLoader": "HDM Loader", "HDMCameraParam": "HDM Camera"}
