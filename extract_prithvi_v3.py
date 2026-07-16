import sys
import json
import numpy as np
import torch
import rasterio
from rasterio.mask import mask as rio_mask
from skimage.transform import resize
import geopandas as gpd
import pickle
from tqdm import tqdm

sys.path.insert(0, "/home/sudiptochakraborty/praktikum_cv/models/prithvi_300m")
from prithvi_mae import PrithviMAE

DATA_DIR   = "/home/sudiptochakraborty/praktikum_cv/Data"
SCENE_DIR  = f"{DATA_DIR}/sentinel2/wuerzburg_core"
FIELDS_PATH = f"{DATA_DIR}/sugar_beet_fields_franconia_hires.geojson"
MODEL_DIR  = "/home/sudiptochakraborty/praktikum_cv/models/prithvi_300m"
CHECKPOINT = f"{MODEL_DIR}/Prithvi_EO_V2_300M.pt"

BANDS  = ["B02", "B03", "B04", "B05", "B06", "B07"]
MEAN   = np.array([1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0])
STD    = np.array([2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0])
DATES  = ["20220416", "20220618", "20220807"]
IMG_SIZE = 64

device = torch.device("cuda")
print(f"Device: {device} — {torch.cuda.get_device_name(0)}")

# ============================================================
# LOAD MODEL
# ============================================================
print("Loading Prithvi EO 2.0 300M...")
with open(f"{MODEL_DIR}/config.json") as f:
    cfg = json.load(f)["pretrained_cfg"]
cfg.update(num_frames=len(DATES), in_chans=6, coords_encoding=[])

model = PrithviMAE(**cfg)
state_dict = torch.load(CHECKPOINT, map_location=device, weights_only=True)
for k in list(state_dict.keys()):
    if 'pos_embed' in k:
        del state_dict[k]
model.load_state_dict(state_dict, strict=False)
model.to(device)
model.eval()
print(f"Model loaded")

# ============================================================
# LOAD + REPROJECT FIELDS
# ============================================================
gdf = gpd.read_file(FIELDS_PATH)
with rasterio.open(f"{SCENE_DIR}/{DATES[0]}/B02.tif") as src:
    scene_crs = src.crs
gdf = gdf.to_crs(scene_crs)
print(f"Fields: {len(gdf)} reprojected to {scene_crs}")

# ============================================================
# HELPER: Crop one band, handle uint16 nodata
# ============================================================
def crop_band(date, band, field_geom):
    path = f"{SCENE_DIR}/{date}/{band}.tif"
    with rasterio.open(path) as src:
        # Use nodata=0 for uint16 (can't use -9999)
        out, _ = rio_mask(src, [field_geom], crop=True,
                          nodata=0, all_touched=True)
        out = out[0].astype(np.float32)
    if out.shape[0] < 2 or out.shape[1] < 2:
        return None
    return out

# ============================================================
# MAIN EXTRACTION
# ============================================================
def extract_embeddings(fields_gdf, label=""):
    results = []
    failed = 0

    for idx, row in tqdm(fields_gdf.iterrows(),
                         total=len(fields_gdf), desc=label):
        field_geom = row.geometry
        field_id   = row["field_id"]

        band_stacks = []
        valid = True

        for b_idx, band in enumerate(BANDS):
            frames = []
            for date in DATES:
                try:
                    patch = crop_band(date, band, field_geom)
                except Exception:
                    valid = False
                    break

                if patch is None:
                    valid = False
                    break

                # Normalize: (reflectance - mean) / std
                patch = (patch - MEAN[b_idx]) / STD[b_idx]

                # Resize to IMG_SIZE x IMG_SIZE
                patch = resize(patch, (IMG_SIZE, IMG_SIZE),
                               anti_aliasing=True,
                               preserve_range=True).astype(np.float32)
                frames.append(patch)

            if not valid:
                break
            band_stacks.append(np.stack(frames, axis=0))  # [T, H, W]

        if not valid or len(band_stacks) != 6:
            failed += 1
            continue

        # [C=6, T=3, H=64, W=64] → [1, C, T, H, W]
        x = np.stack(band_stacks, axis=0)
        x = torch.tensor(x).unsqueeze(0).to(device)

        with torch.no_grad():
            try:
                _, pred, _ = model(x, None, None, 0.0)
                # pred: [1, num_patches, embed_dim]
                # Mean pool → field-level embedding
                emb = pred.mean(dim=1).squeeze(0).cpu().numpy()
            except Exception as e:
                failed += 1
                continue

        results.append({
            "field_id":  field_id,
            "embedding": emb,
            "area_ha":   float(row.get("area_ha", 0)),
        })

    return results, failed

# ============================================================
# SMOKE TEST — 50 FIELDS
# ============================================================
print("\nSmoke test: 50 fields...")
results, failed = extract_embeddings(gdf.head(50), "Smoke test")

print(f"\nSuccessful: {len(results)} / 50")
print(f"Failed: {failed}")
if results:
    print(f"Embedding shape: {results[0]['embedding'].shape}")
    print(f"Embedding sample (first 5 values): {results[0]['embedding'][:5]}")

# Save smoke test
with open(f"{DATA_DIR}/prithvi_embeddings_smoke_v3.pkl", "wb") as f:
    pickle.dump(results, f)

mem = torch.cuda.memory_allocated() / 1024**2
print(f"GPU memory: {mem:.0f} MB / 12288 MB")

if len(results) > 40:
    print("\nSmoke test passed — running full 1020 fields...")
    all_results, all_failed = extract_embeddings(gdf, "Full run")
    print(f"\nFull run: {len(all_results)} / {len(gdf)} successful")
    out_path = f"{DATA_DIR}/prithvi_embeddings_all_fields.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(all_results, f)
    print(f"Saved: {out_path}")
else:
    print("\nSmoke test did not pass threshold — check errors above.")
