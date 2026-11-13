#!/usr/bin/env python
"""Export a fixed-shape ALIKED + LightGlue pair for the local TensorRT 10.8 stack.

This follows the SuperPoint-SuperGlue-TensorRT split: one extractor ONNX and one
matcher ONNX, both with static shapes. Deformable conv is rewritten with
grid_sample so TensorRT can parse it. LightGlue runs every layer with flash,
early-stop and point pruning turned off; mutual-match filtering stays in C++.
"""

import argparse
import sys
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F
import torchvision
from torch import nn

LIGHTGLUE_ROOT = Path("/home/hhl/workspace/algorithm/LightGlue")
sys.path.insert(0, str(LIGHTGLUE_ROOT))

import lightglue.aliked as aliked_mod  # noqa: E402
import lightglue.lightglue as lightglue_mod  # noqa: E402
from lightglue.aliked import ALIKED, simple_nms  # noqa: E402
from lightglue.lightglue import Attention, CrossBlock, LightGlue, SelfBlock, filter_matches, normalize_keypoints  # noqa: E402


def deform_grid(x, offset, weight, bias, padding, align_corners, swap_xy):
    """Bilinear deformable conv expressed with grid_sample + 1x1 kernels."""
    n, _, h, w = x.shape
    oc, _, kh, kw = weight.shape
    if isinstance(padding, (tuple, list)):
        pad_h, pad_w = int(padding[0]), int(padding[1])
    else:
        pad_h = pad_w = int(padding)
    ys = torch.arange(h, device=x.device, dtype=x.dtype)
    xs = torch.arange(w, device=x.device, dtype=x.dtype)
    base_y, base_x = torch.meshgrid(ys, xs, indexing="ij")
    acc = x.new_zeros(n, oc, h, w)
    k = 0
    for iy in range(kh):
        for ix in range(kw):
            c0 = offset[:, 2 * k]
            c1 = offset[:, 2 * k + 1]
            oy, ox = (c1, c0) if swap_xy else (c0, c1)
            sy = base_y + (iy - pad_h) + oy
            sx = base_x + (ix - pad_w) + ox
            if align_corners:
                ny = sy / (h - 1) * 2 - 1
                nx = sx / (w - 1) * 2 - 1
            else:
                ny = (sy + 0.5) / h * 2 - 1
                nx = (sx + 0.5) / w * 2 - 1
            grid = torch.stack((nx, ny), dim=-1)
            sampled = F.grid_sample(
                x, grid, mode="bilinear", padding_mode="zeros", align_corners=align_corners
            )
            acc = acc + F.conv2d(sampled, weight[:, :, iy : iy + 1, ix : ix + 1])
            k += 1
    if bias is not None:
        acc = acc + bias.view(1, -1, 1, 1)
    return acc


def choose_deform_variant(module, device):
    x = torch.randn(1, module.regular_conv.in_channels, 32, 40, device=device)
    with torch.no_grad():
        raw = module.offset_conv(x)
        max_offset = max(x.shape[2], x.shape[3]) / 4.0
        offset = raw.clamp(-max_offset, max_offset)
        ref = torchvision.ops.deform_conv2d(
            x, offset, module.regular_conv.weight, module.regular_conv.bias, padding=module.padding
        )
        best = None
        for align_corners in (True, False):
            for swap_xy in (False, True):
                pred = deform_grid(
                    x, offset, module.regular_conv.weight, module.regular_conv.bias,
                    module.padding, align_corners, swap_xy,
                )
                err = (pred - ref).abs().max().item()
                print(f"deform variant align_corners={align_corners} swap_xy={swap_xy} max_abs={err:.3e}")
                if best is None or err < best[0]:
                    best = (err, align_corners, swap_xy)
    print(
        f"deform_conv grid_sample max_abs={best[0]:.3e} "
        f"align_corners={best[1]} swap_xy={best[2]}"
    )
    if best[0] > 1e-3:
        raise RuntimeError("grid_sample deformable conv does not match torchvision")
    return best[1], best[2]


def install_deform(align_corners, swap_xy):
    def forward(self, x):
        h, w = x.shape[2:]
        limit = x.new_tensor(float(max(h, w)) / 4.0)
        offset = self.offset_conv(x).clamp(-limit, limit)
        return deform_grid(
            x, offset, self.regular_conv.weight, self.regular_conv.bias,
            self.padding, align_corners, swap_xy,
        )

    aliked_mod.DeformableConv2d.forward = forward


def disable_inplace(module):
    for child in module.modules():
        if isinstance(child, (nn.SELU, nn.ReLU, nn.GELU)):
            if hasattr(child, "inplace"):
                child.inplace = False


def dkd_topk(dkd, scores_map):
    """Fixed-K DKD. Same math as ALIKED, without Python lists."""
    _, _, _, width = scores_map.shape
    radius = dkd.radius
    nms = simple_nms(scores_map, radius)
    keep = torch.ones_like(nms)
    keep[:, :, :radius, :] = 0
    keep[:, :, :, :radius] = 0
    keep[:, :, -radius:, :] = 0
    keep[:, :, :, -radius:] = 0
    nms = nms * keep
    flat = nms.view(nms.shape[0], -1)
    indices = torch.topk(flat, dkd.top_k).indices
    peaks = torch.gather(flat, 1, indices)
    patches = dkd.unfold(scores_map)
    gather_index = indices.unsqueeze(1).expand(-1, patches.shape[1], -1)
    patch_scores = torch.gather(patches, 2, gather_index).permute(0, 2, 1)
    xs = (indices % width).to(scores_map.dtype)
    ys = torch.div(indices, width, rounding_mode="trunc").to(scores_map.dtype)
    kpts = torch.stack((xs, ys), dim=-1)
    max_v = patch_scores.max(dim=-1, keepdim=True).values
    weights = ((patch_scores - max_v) / dkd.temperature).exp()
    hw_grid = dkd.hw_grid.to(device=scores_map.device, dtype=scores_map.dtype)
    residual = weights @ hw_grid / weights.sum(dim=-1, keepdim=True)
    kpts = kpts + residual
    wh = scores_map.new_tensor([width - 1, scores_map.shape[2] - 1])
    kpts_norm = kpts / wh * 2 - 1
    sampled = F.grid_sample(
        scores_map, kpts_norm.view(scores_map.shape[0], 1, -1, 2),
        mode="bilinear", align_corners=True,
    )
    return kpts_norm, sampled[:, 0, 0, :], peaks


def sddh_tensor(head, feature_map, kpts_norm):
    """Vectorized SDDH for a single image. kpts_norm is 1xNx2 in [-1, 1]."""
    _, channels, height, width = feature_map.shape
    wh = feature_map.new_tensor([width - 1, height - 1])
    max_offset = float(max(height, width) / 4.0)
    kpts_wh = (kpts_norm[0] / 2 + 0.5) * wh
    count = kpts_wh.shape[0]
    kernel = head.kernel_size
    # ALIKED truncates the pixel coordinate before computing the patch corner.
    corner = (kpts_wh.long().to(kpts_wh.dtype) - kernel / 2 + 1).long()
    corner_x = corner[:, 0].clamp(corner.new_tensor(0), corner.new_tensor(width - 1 - kernel))
    corner_y = corner[:, 1].clamp(corner.new_tensor(0), corner.new_tensor(height - 1 - kernel))
    offset = torch.arange(kernel, device=feature_map.device)
    row = offset.view(kernel, 1, 1)
    col = offset.view(1, kernel, 1)
    ys = corner_y.view(1, 1, count) + row
    xs = corner_x.view(1, 1, count) + col
    patch = feature_map[0][:, ys, xs].permute(3, 0, 1, 2)
    raw_offset = head.offset_conv(patch).clamp(patch.new_tensor(-max_offset), patch.new_tensor(max_offset))
    raw_offset = raw_offset[:, :, 0, 0].view(count, 2, head.n_pos).permute(0, 2, 1)
    pos = kpts_wh.unsqueeze(1) + raw_offset
    pos_norm = (2.0 * pos / wh - 1).reshape(1, count * head.n_pos, 1, 2)
    sampled = F.grid_sample(feature_map, pos_norm, mode="bilinear", align_corners=True)
    sampled = sampled.reshape(channels, count, head.n_pos, 1).permute(1, 0, 2, 3)
    sampled = F.selu(head.sf_conv(sampled)).squeeze(-1)
    desc = torch.einsum("ncp,pcd->nd", sampled, head.agg_weights)
    return F.normalize(desc, p=2, dim=1).unsqueeze(0)


class AlikedExport(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        feature_map, score_map = self.model.extract_dense_map(image)
        kpts_norm, scores, peaks = dkd_topk(self.model.dkd, score_map)
        descriptors = sddh_tensor(self.model.desc_head, feature_map, kpts_norm)
        _, _, height, width = image.shape
        wh = image.new_tensor([width - 1, height - 1])
        keypoints = wh * (kpts_norm + 1) / 2
        return keypoints, scores, descriptors, peaks


def install_export_kernels():
    """Replace ops that this PyTorch ONNX exporter lowers incorrectly."""

    def rotate_half_export(x):
        # q/k are (B, H, N, D). Keep every axis explicit; negative axes are
        # lowered with the exporter's stale rank and TensorRT then rejects them.
        batch, heads, length, dim = x.shape
        pairs = x.reshape(batch, heads, length, dim // 2, 2)
        rotated = torch.cat(( -pairs[:, :, :, :, 1:2], pairs[:, :, :, :, 0:1]), dim=4)
        return rotated.reshape(batch, heads, length, dim)

    def posenc_forward(self, x):
        projected = self.Wr(x)
        batch, length, freq = projected.shape
        emb = torch.stack((torch.cos(projected), torch.sin(projected)), dim=0)
        emb = emb.unsqueeze(2).repeat_interleave(2, dim=4)
        return emb.reshape(2, batch, 1, length, freq * 2)

    lightglue_mod.rotate_half = rotate_half_export
    lightglue_mod.LearnableFourierPositionalEncoding.forward = posenc_forward

    def attn_forward(self, q, k, v, mask=None):
        del mask
        scale = q.shape[-1] ** -0.5
        sim = torch.matmul(q, k.permute(0, 1, 3, 2)) * scale
        attn = torch.softmax(sim, dim=3)
        return torch.matmul(attn, v)

    def self_forward(self, x, encoding, mask=None):
        del mask
        batch, length, _ = x.shape
        heads = self.num_heads
        dim = self.head_dim
        qkv = self.Wqkv(x).reshape(batch, length, heads, dim, 3).permute(0, 2, 1, 3, 4)
        # Reshape drops the trailing singleton that the ONNX exporter keeps, so
        # the following softmax axis stays inside the real tensor rank.
        q = lightglue_mod.apply_cached_rotary_emb(encoding, qkv[..., 0].reshape(batch, heads, length, dim))
        k = lightglue_mod.apply_cached_rotary_emb(encoding, qkv[..., 1].reshape(batch, heads, length, dim))
        v = qkv[..., 2].reshape(batch, heads, length, dim)
        context = self.inner_attn(q, k, v)
        message = self.out_proj(context.permute(0, 2, 1, 3).reshape(batch, length, heads * dim))
        return x + self.ffn(torch.cat((x, message), dim=2))

    def cross_forward(self, x0, x1, mask=None):
        del mask
        batch, n, channels = x0.shape
        m = x1.shape[1]
        heads = self.heads
        dim = channels // heads

        def split(tensor, length):
            return tensor.reshape(batch, length, heads, dim).permute(0, 2, 1, 3)

        qk0 = split(self.to_qk(x0), n) * self.scale**0.5
        qk1 = split(self.to_qk(x1), m) * self.scale**0.5
        v0 = split(self.to_v(x0), n)
        v1 = split(self.to_v(x1), m)
        sim = torch.matmul(qk0, qk1.permute(0, 1, 3, 2))
        attn01 = torch.softmax(sim, dim=3)
        attn10 = torch.softmax(sim.permute(0, 1, 3, 2), dim=3)
        m0 = torch.matmul(attn01, v1).permute(0, 2, 1, 3).reshape(batch, n, channels)
        m1 = torch.matmul(attn10, v0).permute(0, 2, 1, 3).reshape(batch, m, channels)
        m0 = self.to_out(m0)
        m1 = self.to_out(m1)
        x0 = x0 + self.ffn(torch.cat((x0, m0), dim=2))
        x1 = x1 + self.ffn(torch.cat((x1, m1), dim=2))
        return x0, x1

    Attention.forward = attn_forward
    SelfBlock.forward = self_forward
    CrossBlock.forward = cross_forward


def log_assignment_export(layer, desc0, desc1):
    """Same assignment as LightGlue, built without inplace slice writes."""
    mdesc0 = layer.final_proj(desc0)
    mdesc1 = layer.final_proj(desc1)
    scale = float(layer.dim) ** 0.25
    mdesc0 = mdesc0 / scale
    mdesc1 = mdesc1 / scale
    sim = torch.einsum("bmd,bnd->bmn", mdesc0, mdesc1)
    z0 = layer.matchability(desc0)
    z1 = layer.matchability(desc1)
    certainties = F.logsigmoid(z0) + F.logsigmoid(z1).permute(0, 2, 1)
    scores0 = F.log_softmax(sim, dim=2)
    scores1 = F.log_softmax(sim.permute(0, 2, 1).contiguous(), dim=2).permute(0, 2, 1)
    body = scores0 + scores1 + certainties
    dust0 = F.logsigmoid(-z0.squeeze(2))
    dust1 = F.logsigmoid(-z1.squeeze(2))
    top = torch.cat((body, dust0.unsqueeze(2)), dim=2)
    bottom = torch.cat((dust1, dust1.new_zeros(dust1.shape[0], 1)), dim=1).unsqueeze(1)
    return torch.cat((top, bottom), dim=1)


class LightGlueExport(nn.Module):
    def __init__(self, model, width, height):
        super().__init__()
        self.model = model
        self.width = width
        self.height = height

    def forward(self, kpts0, desc0, kpts1, desc1):
        size = kpts0.new_tensor([[self.width, self.height]])
        kpts0 = normalize_keypoints(kpts0, size)
        kpts1 = normalize_keypoints(kpts1, size)
        desc0 = self.model.input_proj(desc0)
        desc1 = self.model.input_proj(desc1)
        enc0 = self.model.posenc(kpts0)
        enc1 = self.model.posenc(kpts1)
        for layer in self.model.transformers:
            desc0, desc1 = layer(desc0, desc1, enc0, enc1)
        return log_assignment_export(self.model.log_assignment[-1], desc0, desc1)


def load_rgb(path, width, height, device):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).to(device=device, dtype=torch.float32).permute(2, 0, 1) / 255.0
    return tensor.unsqueeze(0)


def check_aliked(model, exporter, image):
    with torch.no_grad():
        ref = model({"image": image})
        kpts, scores, desc, _peaks = exporter(image)
    kpt_err = (kpts - ref["keypoints"]).abs().max().item()
    score_err = (scores - ref["keypoint_scores"]).abs().max().item()
    desc_err = (desc - ref["descriptors"]).abs().max().item()
    print(f"ALIKED check max_abs keypoints={kpt_err:.3e} scores={score_err:.3e} desc={desc_err:.3e}")
    if max(kpt_err, score_err, desc_err) > 1e-3:
        raise RuntimeError("vectorized ALIKED does not match the original forward")


def check_threshold_set(exporter, image, ref_keypoints, threshold):
    """Valid peaks must be the same points as ALIKED's threshold detector."""
    with torch.no_grad():
        kpts, _scores, _desc, peaks = exporter(image)
    keep = peaks[0] > threshold
    pred = kpts[0][keep]
    ref = ref_keypoints[0]
    if pred.shape[0] == 0 or ref.shape[0] == 0:
        raise RuntimeError("threshold detector returned no keypoints")
    distance = torch.cdist(pred, ref)
    covered = (distance.min(dim=1).values < 1.0).float().mean().item()
    recalled = (distance.min(dim=0).values < 1.0).float().mean().item()
    print(
        f"threshold set pred={pred.shape[0]} ref={ref.shape[0]} "
        f"covered={covered:.4f} recalled={recalled:.4f}"
    )
    if covered < 0.99 or recalled < 0.99:
        raise RuntimeError("top-k plus threshold does not match the original detector")


def compact_valid(keypoints, descriptors, peaks, threshold):
    keep = peaks[0] > threshold
    return keypoints[:, keep], descriptors[:, keep]


def check_lightglue(matcher, exporter, kpts0, desc0, kpts1, desc1, width, height):
    size = kpts0.new_tensor([[width, height]])
    data = {
        "image0": {"keypoints": kpts0, "descriptors": desc0, "image_size": size},
        "image1": {"keypoints": kpts1, "descriptors": desc1, "image_size": size},
    }
    with torch.no_grad():
        ref = matcher(data)
        scores = exporter(kpts0, desc0, kpts1, desc1)
        m0, _, _, _ = filter_matches(scores, matcher.conf.filter_threshold)
    ref_m = ref["matches0"]
    agree = (m0 == ref_m).float().mean().item()
    print(f"LightGlue check match agreement={agree:.4f} ref_matches={(ref_m > -1).sum().item()}")
    if agree < 0.99:
        raise RuntimeError("static LightGlue does not match the original matcher")


def export_onnx(module, args, path, input_names, output_names, dynamic_axes=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = dict(
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
    )
    if dynamic_axes is not None:
        kwargs["dynamic_axes"] = dynamic_axes
    try:
        torch.onnx.export(module, args, str(path), dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(module, args, str(path), **kwargs)
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    # TensorRT TopK rejects K above 3840. HPatches at 640x480 peaks at 3116 points.
    parser.add_argument("--keypoints", type=int, default=3840)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--skip-threshold-check", action="store_true")
    parser.add_argument("--extractor-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "weights")
    parser.add_argument(
        "--image",
        type=Path,
        default=Path("/home/hhl/workspace/deploy/tensorrt/SuperPoint-SuperGlue-TensorRT/image/image0.png"),
    )
    parser.add_argument(
        "--image2",
        type=Path,
        default=Path("/home/hhl/workspace/deploy/tensorrt/SuperPoint-SuperGlue-TensorRT/image/image1.png"),
    )
    args = parser.parse_args()

    device = torch.device("cuda")
    torch.hub.set_dir(str(args.output_dir / "torchhub"))

    probe = aliked_mod.DeformableConv2d(8, 8, mask=False).to(device).eval()
    align_corners, swap_xy = choose_deform_variant(probe, device)

    image0 = load_rgb(args.image, args.width, args.height, device)
    image1 = load_rgb(args.image2, args.width, args.height, device)
    reference = ALIKED(max_num_keypoints=-1, detection_threshold=args.threshold).to(device).eval()
    with torch.no_grad():
        ref_kpts0 = reference({"image": image0})["keypoints"].detach()
        ref_kpts1 = reference({"image": image1})["keypoints"].detach()
    del reference

    install_deform(align_corners, swap_xy)
    aliked = ALIKED(
        model_name="aliked-n16",
        max_num_keypoints=args.keypoints,
        detection_threshold=-1,
        nms_radius=2,
    ).to(device).eval()
    disable_inplace(aliked)
    extractor = AlikedExport(aliked).to(device).eval()
    check_aliked(aliked, extractor, image0)
    if not args.skip_threshold_check:
        check_threshold_set(extractor, image0, ref_kpts0, args.threshold)
        check_threshold_set(extractor, image1, ref_kpts1, args.threshold)

    export_onnx(
        extractor, (image0,), args.output_dir / f"aliked_n16_k{args.keypoints}.onnx",
        ["image"], ["keypoints", "scores", "descriptors", "peaks"],
    )
    if args.extractor_only:
        return

    matcher = LightGlue(
        features="aliked",
        flash=False,
        mp=False,
        depth_confidence=-1,
        width_confidence=-1,
    ).to(device).eval()
    for module in matcher.modules():
        if isinstance(module, Attention):
            module.enable_flash = False
            module.has_sdp = False
    with torch.no_grad():
        k0, _s0, d0, p0 = extractor(image0)
        k1, _s1, d1, p1 = extractor(image1)
        k0, d0 = compact_valid(k0, d0, p0, args.threshold)
        k1, d1 = compact_valid(k1, d1, p1, args.threshold)
        print(f"LightGlue example points {k0.shape[1]} / {k1.shape[1]}")
        size = k0.new_tensor([[args.width, args.height]])
        reference_matches = matcher({
            "image0": {"keypoints": k0, "descriptors": d0, "image_size": size},
            "image1": {"keypoints": k1, "descriptors": d1, "image_size": size},
        })["matches0"]
    install_export_kernels()
    glue = LightGlueExport(matcher, args.width, args.height).to(device).eval()
    with torch.no_grad():
        exported_matches, _, _, _ = filter_matches(glue(k0, d0, k1, d1), matcher.conf.filter_threshold)
    agreement = (exported_matches == reference_matches).float().mean().item()
    print(
        f"LightGlue check match agreement={agreement:.4f} "
        f"ref_matches={(reference_matches > -1).sum().item()}"
    )
    if agreement < 0.99:
        raise RuntimeError("static LightGlue does not match the original matcher")

    export_onnx(
        glue, (k0, d0, k1, d1), args.output_dir / "aliked_lightglue_dyn.onnx",
        ["kpts0", "desc0", "kpts1", "desc1"], ["scores"],
        dynamic_axes={
            "kpts0": {1: "m"},
            "desc0": {1: "m"},
            "kpts1": {1: "n"},
            "desc1": {1: "n"},
            "scores": {1: "mp1", 2: "np1"},
        },
    )


if __name__ == "__main__":
    main()
