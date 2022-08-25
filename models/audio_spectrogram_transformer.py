# Modified from https://github.com/YuanGongND/ast/blob/master/src/models/ast_models.py

import os
from typing import Any, Tuple

import timm
import torch
import torch.nn as nn
import wget
from timm.models.layers import to_2tuple, trunc_normal_
from torch.cuda.amp import autocast
from torchsummary import summary

os.environ["TORCH_HOME"] = "../../pretrained_models"

MODEL_URLS = {
    "tiny224": "vit_deit_tiny_distilled_patch16_224",
    "small224": "vit_deit_small_distilled_patch16_224",
    "base224": "vit_deit_base_distilled_patch16_224",
    "base384": "vit_deit_base_distilled_patch16_384",
}


class PatchEmbed(nn.Module):
    r"""Override the timm package to relax the input shape constraint."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 64,
    ) -> None:
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class ASTModel(nn.Module):
    r"""
    The AST model.
    :param label_dim: the label dimension, i.e., the number of total classes, it is 527 for AudioSet, 50 for ESC-50, and 35 for speechcommands v2-35
    :param fstride: the stride of patch spliting on the frequency dimension, for 16*16 patchs, fstride=16 means no overlap, fstride=10 means overlap of 6
    :param tstride: the stride of patch spliting on the time dimension, for 16*16 patchs, tstride=16 means no overlap, tstride=10 means overlap of 6
    :param input_fdim: the number of frequency bins of the input spectrogram
    :param input_tdim: the number of time frames of the input spectrogram
    :param imagenet_pretrain: if use ImageNet pretrained model
    :param audioset_pretrain: if use full AudioSet and ImageNet pretrained model
    :param model_size: the model size of AST, should be in [tiny224, small224, base224, base384], base224 and base 384 are same model, but are trained differently during ImageNet pretraining.
    """

    def __init__(
        self,
        label_dim: int = 527,
        fstride: int = 10,
        tstride: int = 10,
        input_fdim: int = 128,
        input_tdim: int = 1024,
        imagenet_pretrain: bool = False,
        audioset_pretrain: bool = False,
        model_size: str = "tiny224",
        verbose: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        assert (
            timm.__version__ == "0.4.5"
        ), "Please use timm == 0.4.5, the code might not be compatible with newer versions."

        if model_size not in MODEL_URLS.keys():
            raise ValueError(
                f"model_size must be one of {MODEL_URLS.keys()}, but got {model_size}"
            )

        if verbose:
            print(f"----------------- AST Model Summary -----------------")
            print(
                f"ImageNet pretraining: {imagenet_pretrain}, AudioSet pretraining: {audioset_pretrain}"
            )
            print(f"Input shape: {input_fdim}*{input_tdim}")

        self.input_fdim = input_fdim
        self.input_tdim = input_tdim
        self.to_latent = nn.Identity()

        timm.models.vision_transformer.PatchEmbed = PatchEmbed

        if not audioset_pretrain:
            self.v = timm.create_model(
                MODEL_URLS[model_size],
                pretrained=imagenet_pretrain,
            )

            self.original_num_patches = self.v.patch_embed.num_patches
            self.oringal_hw = int(self.original_num_patches**0.5)
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(self.original_embedding_dim),
                nn.Linear(self.original_embedding_dim, label_dim),
            )

            f_dim, t_dim = self.get_intermediate_shape(
                fstride, tstride, input_fdim, input_tdim
            )
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches

            if verbose:
                print(f"frequency stride={fstride}, time stride={tstride}")
                print(f"number of patches={num_patches}")

            linear_proj = torch.nn.Conv2d(
                1,
                self.original_embedding_dim,
                kernel_size=(16, 16),
                stride=(fstride, tstride),
            )

            if imagenet_pretrain:
                linear_proj.weight = torch.nn.Parameter(
                    torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1)
                )
                linear_proj.bias = self.v.patch_embed.proj.bias

                new_pos_embed = (
                    self.v.pos_embed[:, 2:, :]
                    .detach()
                    .reshape(
                        1,
                        self.original_num_patches,
                        self.original_embedding_dim,
                    )
                    .transpose(1, 2)
                    .reshape(
                        1,
                        self.original_embedding_dim,
                        self.oringal_hw,
                        self.oringal_hw,
                    )
                )

                if t_dim <= self.oringal_hw:
                    new_pos_embed = new_pos_embed[
                        :,
                        :,
                        :,
                        int(self.oringal_hw / 2)
                        - int(t_dim / 2) : int(self.oringal_hw / 2)
                        - int(t_dim / 2)
                        + t_dim,
                    ]
                else:
                    new_pos_embed = torch.nn.functional.interpolate(
                        new_pos_embed,
                        size=(self.oringal_hw, t_dim),
                        mode="bilinear",
                        align_corners=True,
                    )
                # cut (from middle) or interpolate the first dimension of the positional embedding
                if f_dim <= self.oringal_hw:
                    new_pos_embed = new_pos_embed[
                        :,
                        :,
                        int(self.oringal_hw / 2)
                        - int(f_dim / 2) : int(self.oringal_hw / 2)
                        - int(f_dim / 2)
                        + f_dim,
                        :,
                    ]
                else:
                    new_pos_embed = torch.nn.functional.interpolate(
                        new_pos_embed,
                        size=(f_dim, t_dim),
                        mode="bilinear",
                        align_corners=True,
                    )
                # flatten the positional embedding
                new_pos_embed = new_pos_embed.reshape(
                    1, self.original_embedding_dim, num_patches
                ).transpose(1, 2)
                # concatenate the above positional embedding with the cls token and distillation token of the deit model.
                self.v.pos_embed = nn.Parameter(
                    torch.cat(
                        [self.v.pos_embed[:, :2, :].detach(), new_pos_embed],
                        dim=1,
                    )
                )
            else:
                random_pos_embed = nn.Parameter(
                    torch.zeros(
                        1,
                        self.v.patch_embed.num_patches + 2,
                        self.original_embedding_dim,
                    )
                )
                self.v.pos_embed = random_pos_embed
                trunc_normal_(self.v.pos_embed, std=0.02)

            self.v.patch_embed.proj = linear_proj

        else:
            if not imagenet_pretrain:
                raise ValueError(
                    "currently model pretrained on only audioset is not supported, please set `imagenet_pretrain=True` to use audioset pretrained model."
                )
            if model_size != "base384":
                raise ValueError(
                    "currently only has base384 AudioSet pretrained model."
                )

            assert input_fdim == 128, "currently only support n_mels=128 input."
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            if not os.path.exists("../../pretrained_models/audioset_10_10_0.4593.pth"):
                # this model performs 0.4593 mAP on the audioset eval set
                audioset_mdl_url = "https://www.dropbox.com/s/ca0b1v2nlxzyeb4/audioset_10_10_0.4593.pth?dl=1"
                wget.download(
                    audioset_mdl_url,
                    out="../../pretrained_models/audioset_10_10_0.4593.pth",
                )

            sd = torch.load(
                "../../pretrained_models/audioset_10_10_0.4593.pth", map_location=device
            )

            audio_model = ASTModel(
                label_dim=527,
                fstride=10,
                tstride=10,
                input_fdim=128,
                input_tdim=1024,
                imagenet_pretrain=False,
                audioset_pretrain=False,
                model_size="base384",
                verbose=False,
            )
            audio_model = torch.nn.DataParallel(audio_model)
            audio_model.load_state_dict(sd, strict=False)

            self.v = audio_model.module.v
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(self.original_embedding_dim),
                nn.Linear(self.original_embedding_dim, label_dim),
            )

            f_dim, t_dim = self.get_intermediate_shape(
                fstride, tstride, input_fdim, input_tdim
            )
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches

            if verbose:
                print(f"frequency stride={fstride}, time stride={tstride}")
                print(f"number of patches={num_patches}")

            new_pos_embed = (
                self.v.pos_embed[:, 2:, :]
                .detach()
                .reshape(1, 1212, 768)
                .transpose(1, 2)
                .reshape(1, 768, 12, 101)
            )

            # if the input sequence length is larger than the original audioset (10s), then cut the positional embedding
            if t_dim < 101:
                new_pos_embed = new_pos_embed[
                    :, :, :, 50 - int(t_dim / 2) : 50 - int(t_dim / 2) + t_dim
                ]
            else:
                new_pos_embed = torch.nn.functional.interpolate(
                    new_pos_embed, size=(12, t_dim), mode="bilinear", align_corners=True
                )

            new_pos_embed = new_pos_embed.reshape(1, 768, num_patches).transpose(1, 2)
            self.v.pos_embed = nn.Parameter(
                torch.cat(
                    [self.v.pos_embed[:, :2, :].detach(), new_pos_embed],
                    dim=1,
                )
            )

    def get_intermediate_shape(
        self, fstride: int, tstride: int, input_fdim: int = 128, input_tdim: int = 1024
    ) -> Tuple[int, int]:
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(
            1,
            self.original_embedding_dim,
            kernel_size=(16, 16),
            stride=(fstride, tstride),
        )
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim

    @autocast()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        :param x: the input spectrogram, expected shape: (batch_size, 1, frequency_bins, time_frame_num), e.g., (12, 1, 128, 1024)
        :return: prediction
        """
        B = x.shape[0]
        x = self.v.patch_embed(x)

        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)

        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)

        for blk in self.v.blocks:
            x = blk(x)

        x = self.v.norm(x)
        x = (x[:, 0] + x[:, 1]) / 2
        x = self.to_latent(x)
        x = self.mlp_head(x)
        return x


def ast(
    n_mels: int = 64,
    seq_length: int = 128,
    imgnet_pretrain: bool = True,
    num_classes: int = 4,
    **kwargs: Any,
) -> ASTModel:
    return ASTModel(
        input_fdim=n_mels,
        input_tdim=seq_length,
        imagenet_pretrain=imgnet_pretrain,
        label_dim=num_classes,
        **kwargs,
    )


if __name__ == "__main__":
    input_tdim = 200
    input_fdim = 32
    ast_mdl = ASTModel(
        input_tdim=input_tdim, input_fdim=input_fdim, imagenet_pretrain=True
    )
    test_input = torch.rand([10, 1, input_tdim, input_fdim])
    test_output = ast_mdl(test_input)
    print(test_output.shape)

    input_fdim = 32
    input_tdim = 256
    ast_mdl = ASTModel(
        input_tdim=input_tdim,
        input_fdim=input_fdim,
        label_dim=4,
        audioset_pretrain=False,
        imagenet_pretrain=True,
        model_size="base224",
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_input = torch.rand([10, 1, input_fdim, input_tdim]).to(device)
    ast_mdl = ast_mdl.to(device)

    test_output = ast_mdl(test_input)
    # output should be in shape [10, 4], i.e., 10 samples, each with prediction of 4 classes.
    print(test_output.shape)
    summary(ast_mdl, input_size=(1, input_fdim, input_tdim))
