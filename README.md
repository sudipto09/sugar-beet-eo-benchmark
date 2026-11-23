# Temporal probing of frozen EO foundation models on sugar beet fields

Praktikum project, Computer Vision Lab, University of Würzburg
("AI-Powered Agriculture: Unlocking Potential with Foundation Models").

## Question

Does a frozen Earth-observation foundation model (Prithvi EO 2.0 300M) encode
a crop's temporal behaviour (green-up, senescence, seasonal amplitude of NDVI)
better than a simple spectral baseline, and where in the network does that
information sit?

Nothing is fine-tuned. A ridge-regression probe is fitted on fixed features, so
the score measures what the pretrained model already encodes.

Study area: sugar beet fields around Würzburg (Franconia, Germany),
Sentinel-2, two seasons (2021, 2022), three dates per season.

## Experiments

- **Exp A, layer-wise probing:** which of the 24 encoder blocks carries the
  temporal information.
- **Exp B, perturbation:** shuffle, reverse, or collapse the three dates
  (`repeat_first`) and measure the change in the representation (CKA) and in the
  probe R^2, with paired bootstrap confidence intervals.
- **Exp C, held-out date:** predict the held-out date's value from the other
  dates, Prithvi features vs raw spectral bands, with a paired bootstrap on the
  difference.
- **Resolution test:** native 10-20 m vs a footprint-matched 30 m variant
  (Prithvi was pretrained on 30 m HLS imagery).
- **Clustering benchmark** (`run_seeds_v2.py`): silhouette and Davies-Bouldin,
  field bootstrap with paired differences.

## Main findings so far

- Probe accuracy peaks in an early-to-middle layer plateau (roughly layers 1-6)
  and falls toward the final layer.
- Removing dates destroys the probes, but shuffling or reversing the date order
  barely changes them (CKA >= 0.99): the representation is insensitive to date
  order, which is consistent with the encoder pooling over time rather than
  encoding an ordered trajectory.
- Held-out date (predict the third date from the first two, ridge probe):
  in 2022 Prithvi (R^2 0.277 [0.225, 0.324]) clearly beats NDVI/NDRE alone
  (0.171 [0.122, 0.213]), but not raw spectral bands (0.253 [0.200, 0.297]);
  the paired difference Prithvi - spectral is +0.025 [-0.021, +0.077].
- In 2021 all held-out probes are near zero (R^2 about 0.03-0.04), so that year
  carries little signal; the paired difference is +0.001 [-0.016, +0.019].
  Neither year shows a demonstrable advantage of Prithvi over raw spectral bands.
- Clustering (frozen features, silhouette, 2022): Prithvi about 0.09,
  raw spectral means 0.19, NDVI/NDRE 0.25. Prithvi is worse here, and the paired
  difference to NDVI/NDRE is -0.165 [-0.184, -0.146].
- Best-layer results are robust to the 30 m resampling; the 2022 final layer is
  somewhat resolution-sensitive.

SatMAE and DINOv2 are planned but not run yet.

## Caveats

- DLR CropTypes is a classified (random-forest) product, not ground truth.
- 2022 polygons come from the DLR WMS; 2021 polygons are cut from the V021
  GeoTIFF (class 60 = sugar beet). The WMS uses a different class numbering.
- 2021 (cooler, wetter) and 2022 (hot, dry) are a season contrast, not
  replicates, and the acquisition dates differ by 5-10 days.
- Only encoder features are used. Decoder output must not be reported.
- See [`LIMITATIONS.md`](LIMITATIONS.md) for the full list.

## Repository layout

| Path | Purpose |
|---|---|
| `config.py` | paths, scene dates, per-year polygon lookup |
| `make_polygons_from_croptypes.py`, `extract_fields_highres.py`, `download_dlr_wms.py` | field polygons and label download |
| `download_prithvi_bands.py` | Sentinel-2 band download |
| `compute_targets.py` | NDVI trajectory targets per field |
| `extract_prithvi_v4.py` | encoder features from all 24 blocks; perturbation modes; 30 m variant |
| `probe_temporal.py` | Exp A, B, C; bootstrap and paired bootstrap |
| `compare_variants.py` | native vs 30 m comparison |
| `run_seeds_v2.py` | clustering benchmark |
| `run_year.sh`, `run_variants.sh` | per-year and 30 m pipelines |
| `check_*.py` | data checks (radiometric offset, label class, field purity, NDVI sanity) |
| `tests/` | synthetic data generator |
| `LIMITATIONS.md` | known limitations and open issues |
| `results/` | small result CSVs and `scene_metadata.json` |

Large data (`Data/`, model weights, feature pickles) is not in the repository.

## Running

1. Download the Prithvi EO 2.0 300M weights from Hugging Face
   (`ibm-nasa-geospatial/Prithvi-EO-2.0-300M`) into `models/prithvi_300m/`.
2. Download the Sentinel-2 scenes listed in `config.py` and the DLR CropTypes
   labels into `Data/`.
3. Run the pipeline per year:

```bash
bash run_year.sh 2022
bash run_year.sh 2021 /path/to/CROPTYPES_DE_P1Y_2021_V021.tif
bash run_variants.sh 2022      # 30 m variant; FULL=1 adds perturbation runs
```

Verify the Sentinel-2 radiometric offset first: processing baseline >= 04.00
adds +1000 DN (`check_processing_baseline.py`).

## Evaluation protocol

GEO-Bench style: fixed seeds, interquartile mean, bootstrapped confidence
intervals, paired bootstrap over fields for differences.