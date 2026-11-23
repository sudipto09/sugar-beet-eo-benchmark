"""
Extract frozen Prithvi EO 2.0 encoder features per field.

"""

import argparse
import json
import pickle
import sys

import numpy as np
import torch
from skimage.transform import resize
from tqdm import tqdm

import config

sys.path.insert(0, str(config.MODEL_DIR))

PERTURBATIONS = {
    "none":         "dates in acquisition order (reference)",
    "shuffle":      "date order permuted (fixed permutation)",
    "reverse":      "date order reversed",
    "repeat_first": "first date repeated - all temporal variation removed",
    "mask_last":    "last date replaced by the second - August withheld",
}



# Model loading and encoder API resolution


def load_model(n_frames, device):
    from prithvi_mae import PrithviMAE

    with open(config.MODEL_DIR / "config.json") as fh:
        cfg = json.load(fh)["pretrained_cfg"]
    cfg.update(num_frames=n_frames, in_chans=6, coords_encoding=[])

    model = PrithviMAE(**cfg)
    state = torch.load(config.CHECKPOINT, map_location="cpu", weights_only=True)

    # Positional embeddings are regenerated for this input size.
    dropped = [k for k in state if "pos_embed" in k]
    for key in dropped:
        del state[key]

    missing, unexpected = model.load_state_dict(state, strict=False)
    model.to(device).eval()

    print(f"  checkpoint loaded, {len(dropped)} pos_embed tensors regenerated")
    real_missing = [k for k in missing if "pos_embed" not in k]
    if real_missing:
        print(f"  [warn] {len(real_missing)} weights missing, e.g. "
              f"{real_missing[:3]}")
    if unexpected:
        print(f"  [warn] {len(unexpected)} unexpected keys, e.g. "
              f"{unexpected[:3]}")

    return model


def resolve_encoder(model):
    """Return the ViT encoder module and its transformer block list."""
    encoder = getattr(model, "encoder", model)
    for attr in ("blocks", "layers"):
        blocks = getattr(encoder, attr, None)
        if blocks is not None and len(blocks) > 0:
            return encoder, list(blocks)
    raise RuntimeError(
        "Could not find transformer blocks on the encoder. "
        "Run with --inspect and send me the output.")


def call_encoder(encoder, x):
    """Call the encoder with mask_ratio=0, across known signatures.

    mask_ratio MUST be 0 here. The pretraining default is 0.75, which
    would drop three quarters of the patches and make features that are
    not reproducible between runs.
    """
    attempts = [
        lambda: encoder(x, None, None, 0.0),
        lambda: encoder(x, mask_ratio=0.0),
        lambda: encoder(x, None, None),
        lambda: encoder(x),
    ]
    errors = []
    for attempt in attempts:
        try:
            return attempt()
        except TypeError as exc:
            errors.append(str(exc))
    raise RuntimeError("No encoder signature worked:\n  " +
                       "\n  ".join(errors))


def inspect(model):
    """Print what the local Prithvi copy exposes, for debugging."""
    import inspect as _inspect

    encoder = getattr(model, "encoder", model)
    print("\nPrithviMAE attributes:")
    print("  ", [a for a in dir(model) if not a.startswith("_")][:30])
    print("\nEncoder type:", type(encoder).__name__)
    print("Encoder attributes:")
    print("  ", [a for a in dir(encoder) if not a.startswith("_")][:30])

    for name in ("forward", "forward_features"):
        fn = getattr(encoder, name, None)
        if fn is not None:
            try:
                print(f"\nencoder.{name}{_inspect.signature(fn)}")
            except (ValueError, TypeError):
                print(f"\nencoder.{name}(signature unavailable)")

    try:
        _, blocks = resolve_encoder(model)
        print(f"\nTransformer blocks: {len(blocks)}")
        dim = getattr(encoder, "embed_dim", None)
        print(f"Embedding dim: {dim}")
    except RuntimeError as exc:
        print(f"\n{exc}")



# Feature extraction


class LayerCapture:
    """Capture every transformer block's output with forward hooks.

    Hooks are used rather than forward_features because the latter's
    signature and return type differ between Prithvi releases, while
    block outputs are stable.
    """

    def __init__(self, blocks):
        self.outputs = []
        self._handles = [b.register_forward_hook(self._hook) for b in blocks]

    def _hook(self, _module, _inputs, output):
        if isinstance(output, tuple):
            output = output[0]
        self.outputs.append(output.detach())

    def clear(self):
        self.outputs = []

    def close(self):
        for handle in self._handles:
            handle.remove()


def pool_tokens(hidden, n_expected_patches):
    """[B, T, D] hidden states -> (cls [B, D], mean over patches [B, D])."""
    n_tokens = hidden.shape[1]
    has_cls = n_tokens == n_expected_patches + 1

    if has_cls:
        cls = hidden[:, 0]
        patches = hidden[:, 1:]
    else:
        cls = hidden.mean(dim=1)
        patches = hidden

    return cls.float().cpu().numpy(), patches.mean(dim=1).float().cpu().numpy()


def apply_perturbation(frames, mode, rng=None):
    """Reorder or substitute time steps. `frames` is a list per band.

    `rng` is accepted for API compatibility but no longer used (shuffle
    now applies a fixed derangement rather than a random permutation).
    """
    n = len(frames)
    if mode == "none":
        return list(range(n))
    if mode == "reverse":
        return list(range(n))[::-1]
    if mode == "repeat_first":
        return [0] * n
    if mode == "mask_last":
        order = list(range(n))
        if n >= 2:
            order[-1] = order[-2]
        return order
    if mode == "shuffle":
        # Fixed derangement: every field gets the same non-identity
        # permutation so the perturbation is clean and reproducible.
        # [1, 2, 0] rotates Apr-Jun, Jun-Aug, Aug-Apr.
        if n == 3:
            return [1, 2, 0]
        # Fallback for other frame counts: rotate by 1.
        return [(i + 1) % n for i in range(n)]
    raise ValueError(f"unknown perturbation mode: {mode}")


def pool_to_gsd(patch, native_res, target_gsd):
    """Coarsen a crop to ~target_gsd metre pixels, ignoring invalid pixels.

    `patch` holds NaN where invalid. The crop is area-averaged down by
    native_res / target_gsd using a validity-weighted resize, so a 10 m
    band and a 20 m band both end at the same ground footprint (an integer
    block average would give 30 m for the first and 40 m for the second).
    Used to test the 30 m pretraining resolution against native inputs.
    """
    ratio = native_res / target_gsd
    if ratio >= 1.0:
        return patch
    h, w = patch.shape
    new_h, new_w = max(1, round(h * ratio)), max(1, round(w * ratio))
    valid = np.isfinite(patch).astype(np.float32)
    data = np.where(valid > 0, patch, 0.0).astype(np.float32)
    num = resize(data, (new_h, new_w), anti_aliasing=True, preserve_range=True)
    den = resize(valid, (new_h, new_w), anti_aliasing=True, preserve_range=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        pooled = np.where(den > 1e-3, num / den, np.nan)
    return pooled.astype(np.float32)


def build_patch(gdf_row, dates, offsets, rio_mask, rasterio_mod,
                input_gsd=None, img_size=None):
    """Crop, offset-correct, normalise and resize one field.

    input_gsd : if set (e.g. 30), pool the crop to that pixel size before
                resizing, to match Prithvi's 30 m pretraining resolution.
    img_size  : side length of the model input (default config.IMG_SIZE).

    Returns [C=6, T, H, W] or None if any band is unusable.
    """
    size = img_size or config.IMG_SIZE
    band_stacks = []
    for b_idx, band in enumerate(config.PRITHVI_BANDS_S2):
        frames = []
        for date in dates:
            path = config.SCENE_DIR / date / f"{band}.tif"
            try:
                with rasterio_mod.open(path) as src:
                    out, _ = rio_mask(src, [gdf_row.geometry], crop=True,
                                      nodata=0, all_touched=True)
                    native_res = float(src.res[0])
                patch = out[0].astype(np.float32)
            except Exception:                              # noqa: BLE001
                return None

            if patch.shape[0] < 2 or patch.shape[1] < 2:
                return None

            # Invalid pixels are 0 in the Sentinel-2 L2A DN. They are set to NaN
            invalid = patch <= 0
            patch = patch - offsets[date]
            patch = (patch - config.PRITHVI_MEAN[b_idx]) / config.PRITHVI_STD[b_idx]
            if input_gsd:
                patch[invalid] = np.nan
                patch = pool_to_gsd(patch, native_res, input_gsd)
                patch = np.nan_to_num(patch, nan=0.0)
            else:
                patch[invalid] = 0.0

            patch = resize(patch, (size, size),
                           anti_aliasing=True,
                           preserve_range=True).astype(np.float32)
            frames.append(patch)

        band_stacks.append(np.stack(frames, axis=0))

    return np.stack(band_stacks, axis=0)


def extract(model, gdf, dates, offsets, mode, batch_size, device,
            save_layers, input_gsd=None, img_size=None):
    import geopandas  # noqa: F401  
    import rasterio
    from rasterio.mask import mask as rio_mask

    encoder, blocks = resolve_encoder(model)
    capture = LayerCapture(blocks)
    rng = np.random.default_rng(0)

    patch_grid = (img_size or config.IMG_SIZE) // 16
    n_expected = len(dates) * patch_grid * patch_grid

    results, failed = [], 0
    batch_x, batch_meta = [], []

    def flush():
        nonlocal batch_x, batch_meta
        if not batch_x:
            return
        x = torch.tensor(np.stack(batch_x)).to(device)
        capture.clear()
        with torch.no_grad():
            call_encoder(encoder, x)

        if not capture.outputs:
            raise RuntimeError("No block outputs captured - the encoder "
                               "forward did not run. Use --inspect.")

        per_layer_cls, per_layer_mean = [], []
        for hidden in capture.outputs:
            cls, mean = pool_tokens(hidden, n_expected)
            per_layer_cls.append(cls)
            per_layer_mean.append(mean)

        # [n_layers, B, D] -> [B, n_layers, D]
        layers_mean = np.stack(per_layer_mean, axis=1)
        layers_cls = np.stack(per_layer_cls, axis=1)

        for i, meta in enumerate(batch_meta):
            record = {
                **meta,
                "embedding": layers_mean[i, -1],      # last layer, mean-pooled
                "embedding_cls": layers_cls[i, -1],
            }
            if save_layers:
                record["layers"] = layers_mean[i].astype(np.float16)
            results.append(record)

        batch_x, batch_meta = [], []

    try:
        for _, row in tqdm(gdf.iterrows(), total=len(gdf), desc=f"  {mode}"):
            order = apply_perturbation(dates, mode, rng)
            ordered_dates = [dates[i] for i in order]

            patch = build_patch(row, ordered_dates, offsets, rio_mask, rasterio,
                                input_gsd=input_gsd, img_size=img_size)
            if patch is None:
                failed += 1
                continue

            batch_x.append(patch)
            batch_meta.append({
                "field_id": row["field_id"],
                "area_ha": float(row.get("area_ha", 0.0)),
                "date_order": order,
            })

            if len(batch_x) >= batch_size:
                flush()
        flush()
    finally:
        capture.close()

    return results, failed




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="none", choices=sorted(PERTURBATIONS))
    ap.add_argument("--year", type=int, default=2022)
    ap.add_argument("--limit", type=int, help="smoke test on N fields")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--inspect", action="store_true",
                    help="print the model API and exit")
    ap.add_argument("--no-layers", action="store_true",
                    help="skip per-layer output (smaller files)")
    ap.add_argument("--input-gsd", type=float, default=None,
                    help="pool crops to this pixel size in metres (30 = "
                         "Prithvi pretraining resolution). Use with --tag.")
    ap.add_argument("--img-size", type=int, default=None,
                    help="model input side in pixels, multiple of 16 "
                         "(default config.IMG_SIZE=64; pretraining used 224)."
                         " Use with --tag and a smaller --batch-size.")
    ap.add_argument("--tag", default="",
                    help="suffix for the output file so sensitivity runs do "
                         "not overwrite the baseline, e.g. --tag _gsd30")
    args = ap.parse_args()

    config.describe()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  {torch.cuda.get_device_name(0)}")

    dates = config.DATES[args.year]
    print(f"\nLoading Prithvi EO 2.0 300M ({len(dates)} frames)...")
    model = load_model(len(dates), device)

    if args.inspect:
        inspect(model)
        return

    offsets = {d: config.boa_offset(d) for d in dates}
    print(f"\nYear {args.year}  dates {dates}")
    print(f"Offsets: {', '.join(f'{d}:{o:.0f}' for d, o in offsets.items())}")
    print(f"Mode: {args.mode} - {PERTURBATIONS[args.mode]}")

    import geopandas as gpd
    import rasterio

    gdf = gpd.read_file(config.fields_path(args.year))
    with rasterio.open(config.SCENE_DIR / dates[0] / "B02.tif") as src:
        gdf = gdf.to_crs(src.crs)
    if args.limit:
        gdf = gdf.head(args.limit)
    print(f"Fields: {len(gdf)}\n")

    if args.img_size and args.img_size % 16:
        sys.exit("--img-size must be a multiple of 16")
    if args.input_gsd or args.img_size:
        print(f"Variant: input_gsd={args.input_gsd}  "
              f"img_size={args.img_size or config.IMG_SIZE}  "
              f"tag='{args.tag}'")
        if not args.tag:
            sys.exit("A variant run needs --tag so it does not overwrite "
                     "the baseline pickles.")

    results, failed = extract(model, gdf, dates, offsets, args.mode,
                              args.batch_size, device,
                              save_layers=not args.no_layers,
                              input_gsd=args.input_gsd,
                              img_size=args.img_size)

    print(f"\nExtracted {len(results)} / {len(gdf)}  (failed {failed})")
    if not results:
        print("Nothing extracted - check paths and band files.")
        sys.exit(1)

    emb = results[0]["embedding"]
    print(f"Embedding dim: {emb.shape[0]}  "
          f"(encoder width; 1536 would mean the decoder is still in use)")
    if "layers" in results[0]:
        print(f"Layers saved:  {results[0]['layers'].shape[0]}")
    if emb.shape[0] == 1536:
        print("\n[ERROR] 1536-dim output means decoder patches, not encoder "
              "features. Run --inspect and check resolve_encoder().")

    suffix = "" if args.limit is None else f"_smoke{args.limit}"
    out_path = (config.DATA_DIR /
                f"prithvi_enc_{args.year}_{args.mode}{args.tag}{suffix}.pkl")
    with open(out_path, "wb") as fh:
        pickle.dump(results, fh)
    print(f"\nSaved: {out_path}")

    if device.type == "cuda":
        print(f"GPU peak: "
              f"{torch.cuda.max_memory_allocated() / 1024**2:.0f} MB")


if __name__ == "__main__":
    main()