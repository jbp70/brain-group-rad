"""Per-level groupwise registration network: encoder, linear decoder, heads, centering.

Design sections 4.2-4.4 and 4.7 (`.agents/tasks/network-design/design.md`). Milestone 3 builds
the single-level model for the rigid and affine tasks; the deformable decoder (milestone 6) and
the multi-level cascade (milestone 8) plug into the hooks marked below.

Input: one group per batch, ``images [N, X, Y, Z, 1]`` at one pyramid level (N may be dynamic).

Encoder (identical trainable parameters at every level, so weights copy across levels):

    stem   Conv3D(f0, 3), no norm/act before it    (group conv in config A)
    pool   AveragePooling3D(2), parameter-free     (level 1 only)
    L2     block f1                                (group block in A)
    pool   AveragePooling3D(2), parameter-free     (levels 2 and 1)
    L3     block f2
    L4-L6  blocks f3-f5, stride 2                   -> bottleneck 5x7x6 at every level

Linear decoder (group blocks in both configs; all of it is after the bottleneck):
1x1 reductions (192, 96, 48, 24) -> flatten (5040) -> dense 2048 (group concat only with
``dense_group_concat_first``) -> dense 512 (group) -> norm -> act -> dropout -> heads (float32)
(dense-path norm = ``dense_norm``, LayerNorm by default; the pre-head act is dropped with
``head_activation='none'``):
rotation+translation Dense(13) (zero kernel, identity bias) and scale/shear Dense(6) (zeros).
Rigid: A_raw = [R | t]; affine: A_raw = [R K | t]. A = GroupCentering(A_raw).

Deformable decoder (design 4.5-4.6, milestone 6; built only with ``deform_decoder=True``):
U-Net stages D5..D1, each [trilinear x2 up] -> concat(skip) -> one group block (both configs),
upsamples parameter-free (D5-D3 always, D2 at levels 2/1, D1 at level 1), so the decoder has the
same variables at every level. SVF head: norm -> act -> Conv3D(3, 3) zero-init, float32 ->
v_raw (level voxels, central space). Task 'deform': v = center_velocities(v_raw) and
(u, u_inv) = exp_and_inverse(v, integration_steps) (`models.diffeo`). The inputs are assumed to be
in central space already (A = I). Milestone-8 cascade hook: `models.diffeo.compose_cascade` with
the previous level's fields upsampled by `diffeo.upsample_displacement`.
"""
import collections
import contextlib

import numpy as np
import tensorflow as tf

from data_loading import params as P
from models import checkpointing
from models import diffeo
from models import layers as L
from models import precision
from models import transforms as tfm
from models.config import ModelConfig

TASKS = ("rigid", "affine")
DEFORM_TASK = "deform"
ALL_TASKS = TASKS + (DEFORM_TASK,)
ENCODER_LAYERS = ("stem", "l2", "l3", "l4", "l5", "l6")
DECODER_STAGES = ("d5", "d4", "d3", "d2", "d1")
GRADIENT_CHECKPOINTING = ("none", "integrator", "decoder", "both", "full")
FP16_ACT_CLIP_SCOPES = ("deform_decoder", "all")          # nonfinite-fixes a1
FP16_ACT_CLIP_MAX = 32000.0     # two clipped split-conv terms sum to <= 64000 < 65504
DEFORM_DECODER_DTYPES = ("policy", "float16", "float32", "float32_fine")  # nonfinite-fixes a2
FLOAT32_FINE_STAGES = ("d2", "d1")   # fp32-decoder-width D1: float32 stages of 'float32_fine'
BOTTLENECK_SHAPE = (5, 7, 6)


def _conv_block(cfg, filters, strides, group, name, kernel_size=None, residual_ok=True):
    common = dict(kernel_size=cfg.kernel_size if kernel_size is None else kernel_size,
                  strides=strides, group=group, norm=cfg.norm,
                  negative_slope=cfg.negative_slope, bn_momentum=cfg.bn_momentum,
                  bn_epsilon=cfg.bn_epsilon, seed=cfg.seed, name=name)
    if residual_ok and cfg.block == "residual":
        return L.GroupResBlock(filters, **common)
    return L.GroupConvBlock(filters, **common)


def _n_params(variables):
    return int(sum(int(np.prod(v.shape)) for v in variables))


class GroupEncoder(tf.keras.layers.Layer):
    """Shared encoder for one pyramid level (design 4.3). Returns (bottleneck, skips).

    ``skips`` = [stem, L2, L3, L4, L5, L6] outputs (L6 is the bottleneck), for the deformable
    decoder. Config A: the stem and L2-L6 are group blocks. Config B: no group interaction in
    the encoder, so each image's features depend on that image only (in inference mode; with
    training-mode BatchNorm the batch statistics still couple the group).
    """

    def __init__(self, level, config=None, **kwargs):
        super().__init__(**kwargs)
        self.level = P.check_level(level)
        self.config = config or ModelConfig()
        cfg = self.config
        g = cfg.group_config == "A"
        f = cfg.filters
        self.stem = L.GroupConvBlock(f[0], kernel_size=cfg.kernel_size, group=g,
                                     pre_activation=False, negative_slope=cfg.negative_slope,
                                     seed=cfg.seed, name="stem")
        self.pool2 = tf.keras.layers.AveragePooling3D(2, name="pool2") if self.level == 1 else None
        self.l2 = _conv_block(cfg, f[1], 1, g, "l2")
        self.pool3 = (tf.keras.layers.AveragePooling3D(2, name="pool3")
                      if self.level in (2, 1) else None)
        self.l3 = _conv_block(cfg, f[2], 1, g, "l3")
        self.l4 = _conv_block(cfg, f[3], 2, g, "l4")
        self.l5 = _conv_block(cfg, f[4], 2, g, "l5")
        self.l6 = _conv_block(cfg, f[5], 2, g, "l6")

    def call(self, x, training=None):
        x = self.stem(x)
        skips = [x]
        if self.pool2 is not None:
            x = self.pool2(x)
        x = self.l2(x, training=training)
        skips.append(x)
        if self.pool3 is not None:
            x = self.pool3(x)
        for layer in (self.l3, self.l4, self.l5, self.l6):
            x = layer(x, training=training)
            skips.append(x)
        return x, skips

    def blocks(self):
        return [getattr(self, n) for n in ENCODER_LAYERS]


class LinearDecoder(tf.keras.layers.Layer):
    """Rigid/affine decoder and heads (design 4.4). call(bottleneck, task) -> (A_raw, params, feats).

    Pre-activation blocks throughout: the flattened vector is the raw output of the last 1x1
    reduction conv, and each dense block normalizes and activates its input. A final norm -> act
    after the last dense layer feeds dropout and the heads (the rigid project's Dense -> act ->
    dropout -> head, in pre-activation form). The dense blocks and that pre-head norm-act use
    ``config.dense_norm`` (default 'layer', per-vector LayerNorm; design 4.4, v3.4).

    Heads are float32 (Dense dtype='float32') so the matrix math stays float32 under
    mixed_float16. Translations are in mm: t = translation_scale * raw output (default 1).
    """

    def __init__(self, config=None, **kwargs):
        super().__init__(**kwargs)
        self.config = config or ModelConfig()
        cfg = self.config
        self.reductions = [_conv_block(cfg, r, 1, True, "reduction_%d" % i, kernel_size=1,
                                       residual_ok=False)
                           for i, r in enumerate(cfg.reductions)]
        self.flatten = tf.keras.layers.Flatten(name="flatten")
        self.dense_blocks = [
            L.GroupDenseBlock(u, group=(i > 0 or cfg.dense_group_concat_first),
                              norm=cfg.dense_norm, negative_slope=cfg.negative_slope,
                              bn_momentum=cfg.bn_momentum, bn_epsilon=cfg.bn_epsilon,
                              seed=cfg.seed, name="dense_%d" % i)
            for i, u in enumerate(cfg.dense)]
        self.head_norm_act = L.NormAct(cfg.dense_norm, cfg.negative_slope, cfg.bn_momentum,
                                       cfg.bn_epsilon, activation=cfg.head_activation != "none",
                                       name="head_norm_act")
        self.dropout = tf.keras.layers.Dropout(cfg.dropout, seed=cfg.seed, name="dropout")
        self.rigid_head = tf.keras.layers.Dense(
            tfm.RIGID_PARAMS, kernel_initializer="zeros",
            bias_initializer=tf.keras.initializers.Constant(tfm.identity_rigid_params()),
            dtype="float32", name="rigid_head")
        self.scale_shear_head = tf.keras.layers.Dense(
            tfm.SCALE_SHEAR_PARAMS, kernel_initializer="zeros",
            bias_initializer=tf.keras.initializers.Constant(tfm.identity_scale_shear_params()),
            dtype="float32", name="scale_shear_head")
        scale = np.ones(tfm.RIGID_PARAMS, np.float32)
        scale[tfm.ROTATION_DIMS:] = float(cfg.translation_scale)
        self._param_scale = scale

    def build(self, input_shape):
        # Build both heads now so the variable set does not depend on which task ran first
        # (weight copying, parameter counts, checkpoints).
        units = self.config.dense[-1]
        for head in (self.rigid_head, self.scale_shear_head):
            with tf.name_scope(head.name):
                head.build(tf.TensorShape([None, units]))
        super().build(input_shape)

    def call(self, x, task="rigid", training=None):
        if task not in TASKS:
            raise ValueError("task must be one of {}, got {!r}".format(TASKS, task))
        for block in self.reductions:
            x = block(x, training=training)
        x = self.flatten(x)
        for block in self.dense_blocks:
            x = block(x, training=training)
        feat = self.head_norm_act(x, training=training)
        h = self.dropout(feat, training=training)
        p_rigid = self.rigid_head(h)
        if self.config.translation_scale != 1.0:
            p_rigid = p_rigid * tf.constant(self._param_scale)
        if task == "rigid":
            params = p_rigid
            a_raw = tfm.rigid_matrix(params)
        else:
            params = tf.concat([p_rigid, self.scale_shear_head(h)], axis=-1)
            a_raw = tfm.affine_matrix(params)
        return a_raw, params, {"dense": feat, "pre_head": feat, "head_input": h}


class DeformDecoder(tf.keras.layers.Layer):
    """U-Net decoder + SVF head (design 4.5). call(skips, training) -> (v_raw, dec_feat).

    Input L6 (skips[5]); stage d5..d1 = [TrilinearUpsample2x] -> concat(skip) -> group block
    (pre-activation norm -> act -> concat(group mean) -> conv, in both configs A and B), skips
    L5, L4, L3, L2, stem. Upsamples: d5, d4, d3 at every level; d2 at levels 2 and 1 (mirrors
    the encoder's pool3); d1 at level 1 (mirrors pool2). All upsamples are parameter-free, so the
    variables are identical at every level. The SVF head is NormAct -> Conv3D(3, 3, 'same') with
    zero kernel and bias in float32: v_raw = 0 at init (identity map). dec_feat = d1 output.

    Per-stage dtype (fp32-decoder-width D1): the stages named in ``float32_stages`` (a subset of
    DECODER_STAGES; GroupRegLevelModel passes FLOAT32_FINE_STAGES = d2, d1 for
    deform_decoder_dtype 'float32_fine') are built with their upsample under a float32 policy, and
    then svf_norm_act too; the other stages and this layer keep the policy at construction time.
    `call` casts x to each stage block's compute dtype before its upsample and to svf_norm_act's
    before the head (no-op casts when the dtypes match, so a uniform-dtype decoder's graph is
    unchanged). Variables are float32 and identical in every case.
    """

    def __init__(self, level, config=None, float32_stages=(), **kwargs):
        super().__init__(**kwargs)
        self.level = P.check_level(level)
        self.config = config or ModelConfig()
        float32_stages = tuple(float32_stages)
        bad = [s for s in float32_stages if s not in DECODER_STAGES]
        if bad:
            raise ValueError("float32_stages must be a subset of {}, got {!r}".format(
                DECODER_STAGES, float32_stages))
        self.float32_stages = float32_stages
        cfg = self.config
        widths = cfg.resolved_decoder_widths()
        has_up = {"d5": True, "d4": True, "d3": True, "d2": self.level in (2, 1),
                  "d1": self.level == 1}
        for name, width in zip(DECODER_STAGES, widths):
            scope = (precision.policy_scope("float32") if name in float32_stages
                     else contextlib.nullcontext())
            with scope:
                setattr(self, "up_" + name,
                        L.TrilinearUpsample2x(name="up_" + name) if has_up[name] else None)
                setattr(self, name, _conv_block(cfg, width, 1, True, name, residual_ok=False))
        scope = (precision.policy_scope("float32") if float32_stages
                 else contextlib.nullcontext())
        with scope:
            self.svf_norm_act = L.NormAct(cfg.norm, cfg.negative_slope, cfg.bn_momentum,
                                          cfg.bn_epsilon, name="svf_norm_act")
        self.svf_head = tf.keras.layers.Conv3D(
            3, 3, padding="same", use_bias=True, kernel_initializer="zeros",
            bias_initializer="zeros", dtype="float32", name="svf_head")

    def call(self, skips, training=None):
        skips = list(skips)
        if len(skips) != len(ENCODER_LAYERS):
            raise ValueError("expected {} encoder skips, got {}".format(len(ENCODER_LAYERS),
                                                                        len(skips)))
        # No-op when the dtypes match (default); float32 under --deform-decoder-dtype float32.
        x = tf.cast(skips[5], self.compute_dtype)
        for name, skip in zip(DECODER_STAGES, (skips[4], skips[3], skips[2], skips[1], skips[0])):
            up, block = getattr(self, "up_" + name), getattr(self, name)
            # No-op unless this stage runs in another dtype ('float32_fine': d2, d1 in float32).
            x = tf.cast(x, block.compute_dtype)
            if up is not None:
                x = up(x)
            x = block(tf.concat([x, tf.cast(skip, x.dtype)], axis=-1), training=training)
        dec_feat = x
        x_head = tf.cast(x, self.svf_norm_act.compute_dtype)
        v_raw = self.svf_head(self.svf_norm_act(x_head, training=training))
        return tf.cast(v_raw, tf.float32), dec_feat

    def blocks(self):
        return [getattr(self, n) for n in DECODER_STAGES]


class GroupRegLevelModel(tf.keras.Model):
    """Encoder + linear decoder + in-graph group centering for one pyramid level.

    call(images, task='rigid'|'affine', training) -> dict with
      A_raw [N, 4, 4] float32: raw head matrices (image i -> central space, before centering);
      A [N, 4, 4] float32: GroupCentering(A_raw) = mu^-1 A_raw_i;
      params [N, 13 | 19] float32; bottleneck [N, 5, 7, 6, f5]; skips (list, encoder outputs);
      dense and pre_head (the same tensor: head_norm_act output, pre-dropout) and head_input
      (post-dropout).
    Images are cast to the compute dtype first (float16 under mixed_float16).

    call(images, task='deform', training, return_warped=False) (needs ``deform_decoder=True``)
    -> dict with v_raw (SVF head output), v (group-centered), u = exp(v), u_inv = exp(-v), all
    float32 [N, X, Y, Z, 3] in level voxels; dec_feat (d1 output), bottleneck, skips; plus
    warped = diffeo.warp(float32 images, u) with ``return_warped``. The images must already be
    in central space (A = I). The linear decoder does not run for 'deform' (its variables get
    no gradient), and the deform decoder does not run for the linear tasks.

    gradient_checkpointing ('none' | 'integrator' | 'decoder' | 'both' | 'full'; a Python
    attribute read at call time, not part of ModelConfig) wraps `diffeo.exp_and_inverse` and/or
    the deform decoder in tf.recompute_grad. Recomputing the decoder in training mode runs its
    BatchNorm layers twice per step, so with 'decoder' / 'both' their moving statistics are
    updated twice (gradients unaffected; kept so the m8 recipe reproduces). 'full' (milestone 9,
    design 3.2; deform task only, linear calls unchanged) recomputes the encoder (returning the 6
    skips), the deform decoder and the integrator as two blocks exp(v) / exp(-v), with
    `models.checkpointing.recompute`: every BN moving statistic is updated exactly once per step,
    from the forward pass. It takes the recompute path only once the encoder and the deform
    decoder are built (the build pass runs directly).

    lean_sampler (Python attribute, default False; milestone 9 design 3.3): the integrator uses
    the lean trilinear sampler (`models.layers.sample_voxel_coords(lean=True)`) in every
    checkpointing mode. split_group_conv (default False; design 3.5): `layers.set_split_group_conv`
    on every pre-activation group block (same variables). Neither is part of ModelConfig.

    Opt-in float16 overflow mitigations (nonfinite-fixes; Python attributes, NOT ModelConfig
    fields, so warm-start / cascade config checks are unaffected; defaults add no op):
      fp16_act_clip C (0 = off, else (0, 32000]): `layers.set_fp16_act_clip` clips the float16
        outputs of every GroupConvBlock (both split-conv terms and their sum) and
        BatchNormWithSeparateAffine to [-C, C]. fp16_act_clip_scope 'deform_decoder' (default:
        d5..d1 + svf_norm_act + the blocks' BN) or 'all' (every such layer of this model, encoder
        and linear decoder too; this also changes the linear tasks). No-op under float32.
      deform_decoder_dtype 'policy' (default) | 'float16' | 'float32' | 'float32_fine':
        'policy' = the global policy; 'float16' = explicit alias of it under mixed_float16 (same
        graph; error under a float32 policy); 'float32' constructs the DeformDecoder (and so all
        its sublayers) under a float32 policy while the global policy is mixed_float16 (the fp16
        skips are cast to float32, the encoder stays float16; ~2x decoder activation memory, not
        expected to fit level 1 at 4 pairs under 40000 MB); 'float32_fine' (fp32-decoder-width
        D1) runs only the full- and half-resolution stages d2, d1 (with up_d2 / up_d1 when they
        exist) and svf_norm_act in float32 (svf_head is float32 in every mode), while the
        DeformDecoder itself, d5..d3 and their upsamples keep mixed_float16. 'float32' and
        'float32_fine' are no-ops under a float32 global policy. Variables are float32 with the
        same names, order and shapes in every mode, so checkpoints, warm start,
        `copy_weights_from`, --cascade-from and resume are compatible across modes. Frozen
        cascade stages (`models.cascade`) are always built float32, whatever this flag is.
      bn_epsilon_override E (0 = off): `layers.set_bn_epsilon` on every BatchNormWithSeparateAffine
        of this model (LayerNorm / InstanceNorm untouched; config.bn_epsilon unchanged).

    Hooks: the linear cascade (design 4.7, milestone 8) warps the inputs by the previous level's
    A^-1 and composes Center(R A_prev) outside this single-level model; the deformable cascade
    composes fields with `models.diffeo.compose_cascade`.
    """

    def __init__(self, level, config=None, gradient_checkpointing="none", lean_sampler=False,
                 split_group_conv=False, fp16_act_clip=0.0, fp16_act_clip_scope="deform_decoder",
                 deform_decoder_dtype="policy", bn_epsilon_override=0.0, **kwargs):
        super().__init__(**kwargs)
        self.level = P.check_level(level)
        self.config = config or ModelConfig()
        if gradient_checkpointing not in GRADIENT_CHECKPOINTING:
            raise ValueError("gradient_checkpointing must be one of {}, got {!r}".format(
                GRADIENT_CHECKPOINTING, gradient_checkpointing))
        fp16_act_clip = float(fp16_act_clip)
        if not (fp16_act_clip == 0.0 or 0.0 < fp16_act_clip <= FP16_ACT_CLIP_MAX):
            raise ValueError("fp16_act_clip must be 0 (off) or in (0, {:g}], got {!r}".format(
                FP16_ACT_CLIP_MAX, fp16_act_clip))
        if fp16_act_clip_scope not in FP16_ACT_CLIP_SCOPES:
            raise ValueError("fp16_act_clip_scope must be one of {}, got {!r}".format(
                FP16_ACT_CLIP_SCOPES, fp16_act_clip_scope))
        if deform_decoder_dtype not in DEFORM_DECODER_DTYPES:
            raise ValueError("deform_decoder_dtype must be one of {}, got {!r}".format(
                DEFORM_DECODER_DTYPES, deform_decoder_dtype))
        global_compute = tf.keras.mixed_precision.global_policy().compute_dtype
        if deform_decoder_dtype == "float16" and global_compute != "float16":
            raise ValueError("deform_decoder_dtype 'float16' needs a float16 global policy "
                             "(mixed_float16), got compute dtype {}".format(global_compute))
        bn_epsilon_override = float(bn_epsilon_override)
        if not (bn_epsilon_override == 0.0 or (np.isfinite(bn_epsilon_override)
                                               and bn_epsilon_override > 0.0)):
            raise ValueError("bn_epsilon_override must be 0 (off) or finite > 0, got {!r}".format(
                bn_epsilon_override))
        if fp16_act_clip > 0 and fp16_act_clip_scope == "deform_decoder" \
                and not self.config.deform_decoder:
            raise ValueError("fp16_act_clip_scope 'deform_decoder' needs "
                             "ModelConfig(deform_decoder=True)")
        self.gradient_checkpointing = gradient_checkpointing
        self.lean_sampler = bool(lean_sampler)
        self.split_group_conv = bool(split_group_conv)
        self.fp16_act_clip = fp16_act_clip
        self.fp16_act_clip_scope = fp16_act_clip_scope
        self.deform_decoder_dtype = deform_decoder_dtype
        self.bn_epsilon_override = bn_epsilon_override
        self.input_shape_level = tuple(P.LEVEL_SHAPE[self.level])
        self.spacing = P.LEVEL_SPACING[self.level]
        self.encoder = GroupEncoder(self.level, self.config, name="encoder")
        self.linear_decoder = LinearDecoder(self.config, name="linear_decoder")
        self.centering = tfm.GroupCentering(iters=self.config.centering_iters, name="centering")
        decoder_scope = (precision.policy_scope("float32")
                         if deform_decoder_dtype == "float32" and global_compute != "float32"
                         else contextlib.nullcontext())
        # 'float32_fine' under a float32 global policy is a no-op (like 'float32').
        float32_stages = (FLOAT32_FINE_STAGES
                          if deform_decoder_dtype == "float32_fine" and global_compute != "float32"
                          else ())
        with decoder_scope:
            self.deform_decoder = (DeformDecoder(self.level, self.config,
                                                 float32_stages=float32_stages,
                                                 name="deform_decoder")
                                   if self.config.deform_decoder else None)
        if self.split_group_conv:
            L.set_split_group_conv(self, True)
        if fp16_act_clip > 0:
            L.set_fp16_act_clip(self.deform_decoder if fp16_act_clip_scope == "deform_decoder"
                                else self, fp16_act_clip)
        if bn_epsilon_override > 0:
            L.set_bn_epsilon(self, bn_epsilon_override)

    def call(self, images, task="rigid", training=None, return_warped=False):
        if task == DEFORM_TASK:
            return self._call_deform(images, training, return_warped)
        x = tf.cast(images, self.compute_dtype)
        bottleneck, skips = self.encoder(x, training=training)
        a_raw, params, feats = self.linear_decoder(bottleneck, task=task, training=training)
        out = {"A_raw": a_raw, "A": self.centering(a_raw), "params": params,
               "bottleneck": bottleneck, "skips": skips}
        out.update(feats)
        return out

    def _call_deform(self, images, training, return_warped):
        if self.deform_decoder is None:
            raise ValueError("task 'deform' needs ModelConfig(deform_decoder=True)")
        mode = self.gradient_checkpointing
        if mode not in GRADIENT_CHECKPOINTING:
            raise ValueError("gradient_checkpointing must be one of {}, got {!r}".format(
                GRADIENT_CHECKPOINTING, mode))
        x = tf.cast(images, self.compute_dtype)
        encoder, decoder = self.encoder, self.deform_decoder
        steps = self.config.integration_steps
        lean = self.lean_sampler
        if mode == "full" and encoder.built and decoder.built:
            return self._call_deform_full(images, x, training, return_warped)
        bottleneck, skips = encoder(x, training=training)
        # Variables must exist before tf.recompute_grad (first call runs the decoder directly).
        if mode in ("decoder", "both") and decoder.built:
            def run_decoder(*s):
                return decoder(list(s), training=training)
            v_raw, dec_feat = tf.recompute_grad(run_decoder)(*skips)
        else:
            v_raw, dec_feat = decoder(skips, training=training)
        v = diffeo.center_velocities(v_raw)
        if mode in ("integrator", "both"):
            def run_integrator(vel):
                return diffeo.exp_and_inverse(vel, steps, lean=lean)
            u, u_inv = tf.recompute_grad(run_integrator)(v)
        else:
            u, u_inv = diffeo.exp_and_inverse(v, steps, lean=lean)
        return self._deform_outputs(images, v_raw, v, u, u_inv, dec_feat, bottleneck, skips,
                                    return_warped)

    def _call_deform_full(self, images, x, training, return_warped):
        """'full' (design 3.2): recomputed encoder (6 skips), recomputed decoder (both with the BN
        statistics updated once), then exp(v) and exp(-v) as two separate recompute blocks."""
        encoder, decoder = self.encoder, self.deform_decoder
        steps = self.config.integration_steps
        lean = self.lean_sampler
        enc_bn = L.batch_norm_layers(encoder)
        dec_bn = L.batch_norm_layers(decoder)

        def run_encoder(inp):
            return encoder(inp, training=training)[1]

        def run_decoder(*s):
            return decoder(list(s), training=training)

        def run_integrator(vel):
            return diffeo.integrate_svf(vel, steps, lean=lean)

        skips = list(checkpointing.recompute(run_encoder, enc_bn, training)(x))
        v_raw, dec_feat = checkpointing.recompute(run_decoder, dec_bn, training)(*skips)
        v = diffeo.center_velocities(v_raw)
        u = tf.recompute_grad(run_integrator)(v)
        u_inv = tf.recompute_grad(run_integrator)(-v)
        return self._deform_outputs(images, v_raw, v, u, u_inv, dec_feat, skips[5], skips,
                                    return_warped)

    def _deform_outputs(self, images, v_raw, v, u, u_inv, dec_feat, bottleneck, skips, return_warped):
        out = {"v_raw": v_raw, "v": v, "u": u, "u_inv": u_inv, "dec_feat": dec_feat,
               "bottleneck": bottleneck, "skips": skips}
        if return_warped:
            out["warped"] = diffeo.warp(tf.cast(images, tf.float32), u)
        return out

    # ------------------------------------------------------------------ utilities
    def build_model(self, n=1):
        """Create every variable with one inference pass on zeros [n, X, Y, Z, 1]. Returns self.

        Runs the rigid task (encoder, both linear heads) and, when the deform decoder exists, one
        deform forward too.
        """
        x = tf.zeros((n,) + self.input_shape_level + (1,), tf.float32)
        self(x, task="rigid", training=False)
        if self.deform_decoder is not None:
            self(x, task=DEFORM_TASK, training=False)
        return self

    def set_bn_stats_frozen(self, frozen=True):
        """Freeze (or unfreeze) every BatchNorm's statistics; gamma/beta stay trainable."""
        return L.set_bn_stats_frozen(self, frozen)

    @staticmethod
    def _local_name(v):
        # Drop the (uniquified) top-level model scope: "group_reg_level_model_3/encoder/..."
        return v.name.split("/", 1)[1] if "/" in v.name else v.name

    def copy_weights_from(self, other):
        """Copy every variable (trainable and BN statistics) from a model of another level.

        Both models must be built and have the same config. Variables are matched by order and
        checked by shape and scope-local name. Returns the number of variables copied.
        """
        # Shape-neutral fields (translation_scale, negative_slope, dropout, centering_iters, ...)
        # would copy silently and change the function, so the configs must match exactly. The
        # init seed is irrelevant once weights are copied.
        mine = {k: v for k, v in self.config.to_dict().items() if k != "seed"}
        theirs = {k: v for k, v in other.config.to_dict().items() if k != "seed"}
        if mine != theirs:
            diff = sorted(k for k in mine if mine[k] != theirs.get(k))
            raise ValueError("model configs differ in {}; build both levels from one ModelConfig".format(diff))
        src, dst = other.weights, self.weights
        if not src or not dst:
            raise ValueError("build both models first (build_model)")
        if len(src) != len(dst):
            raise ValueError("variable count differs: {} vs {}".format(len(src), len(dst)))
        for s, d in zip(src, dst):
            if tuple(s.shape) != tuple(d.shape) or self._local_name(s) != self._local_name(d):
                raise ValueError("variable mismatch: {} {} vs {} {}".format(
                    s.name, s.shape, d.name, d.shape))
        for s, d in zip(src, dst):
            d.assign(s)
        return len(src)

    def count_params_by_part(self):
        """Trainable parameter counts per part (plus group-concat overhead and totals)."""
        enc, dec = self.encoder, self.linear_decoder
        out = collections.OrderedDict()
        for name, block in zip(ENCODER_LAYERS, enc.blocks()):
            out["encoder/" + name] = _n_params(block.trainable_weights)
        out["encoder"] = _n_params(enc.trainable_weights)
        for block in dec.reductions:
            out["linear_decoder/" + block.name] = _n_params(block.trainable_weights)
        for block in dec.dense_blocks:
            out["linear_decoder/" + block.name] = _n_params(block.trainable_weights)
            out["linear_decoder/" + block.name + "/dense"] = _n_params(block.dense.trainable_weights)
        for layer in (dec.head_norm_act, dec.rigid_head, dec.scale_shear_head):
            out["linear_decoder/" + layer.name] = _n_params(layer.trainable_weights)
        out["linear_decoder"] = _n_params(dec.trainable_weights)
        out["group_overhead/encoder"] = int(sum(b.group_overhead_params() for b in enc.blocks()))
        out["group_overhead/linear_decoder"] = int(
            sum(b.group_overhead_params() for b in list(dec.reductions) + list(dec.dense_blocks)))
        ddec = self.deform_decoder
        if ddec is not None:
            for name, block in zip(DECODER_STAGES, ddec.blocks()):
                out["deform_decoder/" + name] = _n_params(block.trainable_weights)
            out["deform_decoder/svf_head"] = _n_params(ddec.svf_norm_act.trainable_weights
                                                       + ddec.svf_head.trainable_weights)
            out["deform_decoder"] = _n_params(ddec.trainable_weights)
            out["group_overhead/deform_decoder"] = int(
                sum(b.group_overhead_params() for b in ddec.blocks()))
        out["trainable"] = _n_params(self.trainable_weights)
        out["non_trainable"] = _n_params(self.non_trainable_weights)
        return out
