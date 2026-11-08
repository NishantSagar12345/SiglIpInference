"""Token-reduced forward path for the pretrained SigLIP vision tower.

What the installed Transformers implementation does (checked against
transformers 4.46 and 5.x source of ``modeling_siglip.py``):

* ``SiglipVisionEmbeddings.forward``: Conv2d(16x16, stride 16) patchify ->
  flatten to (B, 196, 768) in row-major order -> add a *learned absolute*
  position embedding, one vector per patch (``position_embedding(position_ids)``).
  There is no CLS token.
* ``SiglipEncoder.forward(inputs_embeds, attention_mask=None)``: 12 pre-LN
  transformer layers. No attention mask is used for images, and nothing in the
  layers depends on the sequence length being 196.
* ``post_layernorm`` is applied per token.
* ``SiglipMultiheadAttentionPoolingHead``: a learned probe query attends over
  *all* tokens (``nn.MultiheadAttention``), followed by LayerNorm + MLP. It works
  for any number of key/value tokens.

Consequently a reduced sequence can be produced *after* the pretrained
embedding layer has added positional information, and fed through the
unmodified pretrained encoder, post-LayerNorm and pooling head. No weights,
layers or image resolution are changed; each retained token keeps exactly the
position embedding of the patch it came from.

Module layout differs between versions:
* transformers 4.x: ``SiglipVisionModel.vision_model`` is a ``SiglipVisionTransformer``
  holding ``embeddings / encoder / post_layernorm / head``.
* transformers 5.x: ``SiglipVisionModel`` holds those modules directly.
``resolve_vision_core`` handles both.
"""
from __future__ import annotations

import torch
from torch import nn


def resolve_vision_core(model: nn.Module) -> nn.Module:
    """Return the module that owns embeddings/encoder/post_layernorm."""
    candidates = [model]
    if hasattr(model, "vision_model"):
        candidates.insert(0, model.vision_model)
        if hasattr(model.vision_model, "vision_model"):
            candidates.insert(0, model.vision_model.vision_model)
    for c in candidates:
        if all(hasattr(c, a) for a in ("embeddings", "encoder", "post_layernorm")):
            return c
    raise AttributeError(
        "Could not locate SigLIP vision embeddings/encoder/post_layernorm in the "
        f"loaded model ({type(model).__name__}). Inspect modeling_siglip.py for this "
        "transformers version and update resolve_vision_core()."
    )


class TokenReducedSiglipVision(nn.Module):
    """Wraps the pretrained SigLIP vision tower; adds a token-selection step.

    Pipeline::

        pixel_values --embed()--> (B,196,D) tokens (patch + position)
                     --select()-> (B,N,D)  retained tokens, fixed indices
                     --encode()-> last_hidden_state (B,N,D), pooled (B,D)
    """

    def __init__(self, vision_model: nn.Module):
        super().__init__()
        core = resolve_vision_core(vision_model)
        self.embeddings = core.embeddings
        self.encoder = core.encoder
        self.post_layernorm = core.post_layernorm
        self.use_head = bool(getattr(core, "use_head", hasattr(core, "head"))) and hasattr(core, "head")
        self.head = core.head if self.use_head else None

        # SiglipVisionModel carries a SiglipVisionConfig; the full SiglipModel nests it as .vision_config
        cfg = getattr(vision_model.config, "vision_config", None) or vision_model.config
        self.hidden_size = cfg.hidden_size
        self.patch_size = cfg.patch_size
        self.image_size = cfg.image_size
        self.grid_size = cfg.image_size // cfg.patch_size
        self.num_patches = self.grid_size ** 2
        assert self.embeddings.position_embedding.weight.shape == (self.num_patches, self.hidden_size)

    # -- stages ---------------------------------------------------------------
    def embed(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Pretrained embedding layer (patchify + learned position embedding)."""
        assert pixel_values.ndim == 4 and pixel_values.shape[1:] == (3, self.image_size, self.image_size), (
            f"expected (B,3,{self.image_size},{self.image_size}), got {tuple(pixel_values.shape)}"
        )
        tokens = self.embeddings(pixel_values)  # interpolate_pos_encoding=False -> no resizing
        assert tokens.shape == (pixel_values.shape[0], self.num_patches, self.hidden_size), tokens.shape
        return tokens

    @staticmethod
    def select(tokens: torch.Tensor, idx: torch.Tensor | None) -> torch.Tensor:
        """Keep tokens at ``idx`` (same indices for every image in the batch)."""
        if idx is None:
            return tokens
        out = tokens.index_select(1, idx)
        assert out.shape == (tokens.shape[0], idx.numel(), tokens.shape[2]), out.shape
        return out

    def encode(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Pretrained encoder + post-LayerNorm + attention-pooling head."""
        assert tokens.ndim == 3 and tokens.shape[-1] == self.hidden_size, tokens.shape
        hidden = self.encoder(inputs_embeds=tokens)[0]
        assert hidden.shape == tokens.shape, (hidden.shape, tokens.shape)
        hidden = self.post_layernorm(hidden)
        pooled = self.head(hidden) if self.use_head else None
        if pooled is not None:
            assert pooled.shape == (tokens.shape[0], self.hidden_size), pooled.shape
        return hidden, pooled

    def forward(self, pixel_values: torch.Tensor, idx: torch.Tensor | None = None):
        return self.encode(self.select(self.embed(pixel_values), idx))


# -- validation ---------------------------------------------------------------
@torch.inference_mode()
def validate_baseline(model: nn.Module, wrapper: TokenReducedSiglipVision,
                      pixel_values: torch.Tensor, atol: float, rtol: float = 0.0) -> dict:
    """Compare the wrapper's 196-token path against the standard HF forward."""
    ref = model(pixel_values=pixel_values)
    ref_hidden, ref_pooled = ref.last_hidden_state, ref.pooler_output
    identity = torch.arange(wrapper.num_patches, device=pixel_values.device)
    hidden, pooled = wrapper(pixel_values, identity)
    hidden_noidx, pooled_noidx = wrapper(pixel_values, None)

    def diff(a, b):
        return float((a.float() - b.float()).abs().max())

    res = {
        "dtype": str(pixel_values.dtype),
        "batch": int(pixel_values.shape[0]),
        "max_abs_diff_last_hidden": diff(hidden, ref_hidden),
        "max_abs_diff_pooled": diff(pooled, ref_pooled),
        "max_abs_diff_identity_vs_noselect": diff(pooled, pooled_noidx),
        "min_cosine_pooled": float(torch.nn.functional.cosine_similarity(
            pooled.float(), ref_pooled.float(), dim=-1).min()),
        "atol": atol,
    }
    assert hidden.shape == ref_hidden.shape and pooled.shape == ref_pooled.shape
    assert torch.allclose(hidden.float(), ref_hidden.float(), atol=atol, rtol=rtol), res
    assert torch.allclose(pooled.float(), ref_pooled.float(), atol=atol, rtol=rtol), res
    res["passed"] = True
    return res


@torch.inference_mode()
def validate_positional_preservation(wrapper: TokenReducedSiglipVision, pixel_values: torch.Tensor,
                                     idx: torch.Tensor, atol: float) -> dict:
    """Check selected token j == patch_embed[idx[j]] + pos_embed[idx[j]].

    Recomputes the embedding by hand from the conv and position tables, so it
    verifies both that the embedding layer behaves as inspected and that every
    retained token carries the position embedding of its original patch.
    """
    emb = wrapper.embeddings
    patches = emb.patch_embedding(pixel_values.to(emb.patch_embedding.weight.dtype))
    patches = patches.flatten(2).transpose(1, 2)                        # (B,196,D)
    manual = patches[:, idx] + emb.position_embedding.weight[idx][None]  # (B,N,D)
    selected = wrapper.select(wrapper.embed(pixel_values), idx)
    err = float((manual.float() - selected.float()).abs().max())
    assert err <= atol, f"positional preservation check failed: max abs err {err}"
    return {"num_tokens": idx.numel(), "max_abs_err": err, "passed": True}


# -- theoretical cost -----------------------------------------------------------
def analytic_flops(num_tokens: int, cfg) -> dict:
    """Matmul FLOPs (2 x MACs) of encoder + pooling head for N tokens.

    Per layer: QKV+out projections 8*N*D^2, MLP 4*N*D*I, attention
    scores + weighted sum 4*N^2*D. Head: probe attention over N keys plus
    a 1-token MLP. LayerNorm, softmax, GELU and residual adds are omitted.
    The patch-embedding conv (always over all 196 patches) is reported
    separately because it is not reduced by token selection.
    """
    N, D, I, L = num_tokens, cfg.hidden_size, cfg.intermediate_size, cfg.num_hidden_layers
    proj = 8 * N * D * D
    mlp = 4 * N * D * I
    attn = 4 * N * N * D
    head = 2 * D * D + 4 * N * D * D + 4 * N * D + 2 * D * D + 4 * D * I
    patch_embed = cfg.image_size ** 2 // cfg.patch_size ** 2 * 2 * (cfg.num_channels * cfg.patch_size ** 2) * D
    enc = L * (proj + mlp + attn)
    return {
        "num_tokens": N,
        "flops_linear_layers": L * (proj + mlp),
        "flops_attention_quadratic": L * attn,
        "flops_head": head,
        "flops_encoder_plus_head": enc + head,
        "flops_patch_embed_unreduced": patch_embed,
        "quadratic_share": L * attn / (enc + head),
    }
