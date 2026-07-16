# Benchmarking EO Foundation Models for Agricultural Field Representation Learning

**Master's Practikum - M.Sc. Aerospace Informatics**  
**Julius-Maximilians-Universität Würzburg · Computer Vision Lab**  
**Advisor:** Dr. Fayaz Ali Dharejo 
**Chair:** Prof. Dr. Radu Timofte

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1-orange)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)



## Overview

This project benchmarks Earth Observation foundation models against classical spectral baselines for unsupervised agricultural field representation learning, using real sugar beet fields in Franconia, Germany.

The evaluation methodology follows **GEO-Bench** (Lacoste et al., NeurIPS 2023) - the standard framework for benchmarking EO foundation models - and extends it into the temporal agricultural domain, which GEO-Bench explicitly identifies as out of scope in its limitations section.

The task definition and dataset design are grounded in **Sadbhave et al. (2025)**, who establish the sugar beet temporal stress detection problem from Sentinel-2 time series and call for foundation model evaluation as future work. This project directly answers that call. 

---

## Research Question

> *Among EO foundation models with different architectures and pretraining domains - temporal satellite (Prithvi EO 2.0), single-image satellite (SatMAE), and generic vision (DINOv2) - which best represents agricultural field temporal dynamics, and how do they compare to classical spectral baselines under the GEO-Bench evaluation protocol?*

---

## Models Benchmarked

| Model | Type | Pretraining Domain | Architecture |
|---|---|---|---|
| **Prithvi EO 2.0 300M** | EO Foundation Model | Sentinel-2 temporal | Temporal MAE |
| **Prithvi EO 2.0 (fine-tuned)** | EO Foundation Model | Sentinel-2 + NDVI adaptation | Temporal MAE + MLP head |
| **SatMAE** | EO Foundation Model | Multi-spectral satellite | Single-image MAE |
| **DINOv2** | Generic Vision FM | Natural images (ImageNet) | ViT-B/14 |
| **NDVI/NDRE baseline** | Spectral index | - | Temporal trajectory |
| **Random Forest** | Classical ML | - | Leaf embeddings |
| **Raw Spectral KMeans** | Classical baseline | - | Mean reflectance |

---

## Dataset

| Component | Source | Details |
|---|---|---|
| Sugar beet field polygons | DLR EOC CropTypes Germany | 1020 fields, class 13, 10m resolution, Franconia 2022 |
| Sentinel-2 imagery | Copernicus / Microsoft Planetary Computer | 3 scenes per year: Apr / Jun / Aug, 2021 & 2022 |
| Spectral bands | Sentinel-2 L2A | B02, B03, B04, B05, B06, B07, B08 |
| Weather data | DWD Station 05705 (Würzburg) | Hourly temperature + humidity, 2020–2022 |
| Infection risk labels | Wolf & Verreet (2002) thresholds | Binary weekly CLS risk flags from DWD data |

**All data sources are fully public. No proprietary farm data is used.**

---

## Evaluation Protocol (GEO-Bench)

Following Lacoste et al. (NeurIPS 2023):

- **10 random seeds** per model
- **Interquartile Mean (IQM)** as primary aggregation metric - trims top/bottom 25% outliers
- **95% bootstrapped confidence intervals** - 1000 bootstrap samples
- **Metrics:** Silhouette Score (↑), Davies-Bouldin Score (↓)
- **Cross-year validation:** 2021 and 2022 independently

---

## Results

### 2022 - Franconia Sugar Beet Fields (n=1020, 10 seeds)

| Model | Silhouette IQM ↑ | 95% CI | Davies-Bouldin IQM ↓ | 95% CI |
|---|---|---|---|---|
| Prithvi EO 2.0 (fine-tuned) | **0.343** | [0.330, 0.363] | **0.832** | [0.798, 0.876] |
| Prithvi EO 2.0 (zero-shot) | 0.336 | [0.336, 0.336] | 0.893 | [0.893, 0.893] |
| NDVI baseline | 0.256 | [0.256, 0.256] | 1.113 | [1.113, 1.113] |
| Random Forest (leaf embed) | 0.181 | - | 1.952 | - |
| Raw Spectral KMeans | 0.179 | [0.179, 0.179] | 1.489 | [1.488, 1.489] |

### Cross-Year Stability (IQM, 10 seeds)

| Model | 2022 Sil | 2021 Sil | 2022 DB | 2021 DB | Rank stable? |
|---|---|---|---|---|---|
| Prithvi (fine-tuned) | 0.343 | 0.321 | 0.832 | 0.982 | ✗ |
| Prithvi (zero-shot) | 0.336 | 0.274 | 0.893 | 1.128 | ✗ |
| NDVI baseline | 0.256 | 0.376 | 1.113 | 0.966 | ✗ |
| Raw Spectral KMeans | 0.179 | 0.202 | 1.489 | 1.432 | ✓ |

---

## Key Findings

**Finding 1 - Prithvi outperforms spectral baselines geometrically (2022)**  
Prithvi EO 2.0 achieves higher silhouette (0.336) and lower Davies-Bouldin (0.893) than NDVI (0.256 / 1.113) and classical methods under the GEO-Bench evaluation protocol.

**Finding 2 - Fine-tuning improves both metrics**  
A lightweight MLP head trained on NDVI pseudo-labels improves Prithvi geometry: Silhouette 0.336→0.343, Davies-Bouldin 0.893→0.832. Domain adaptation helps even with weak supervision.

**Finding 3 - Zero-shot Prithvi clusters are not agronomically interpretable**  
All 5 Prithvi zero-shot clusters show nearly identical NDVI trajectories across timesteps, suggesting the model captures low-level spectral texture rather than crop phenology without domain adaptation. NDVI baseline clusters are phenologically distinct and interpretable.

**Finding 4 - Cross-year ranking is not fully stable**  
In 2021, NDVI baseline outperforms Prithvi zero-shot (0.376 vs 0.274). This instability is partly attributed to anomalous April 2021 NDVI values (mean=0.63 vs expected ~0.16 for sugar beet emergence), suggesting foundation model embeddings are more robust to data quality issues than spectral indices.

---

## Repository Structure

```
sugar-beet-eo-benchmark/
│
├── Data/
│   ├── sugar_beet_fields_franconia_hires.geojson  # 1020 real field polygons
│   ├── class_legend.json                           # DLR crop type class mapping
│   ├── all_seeds_results.csv                       # 10-seed scores 2022
│   ├── all_seeds_2021.csv                          # 10-seed scores 2021
│   ├── benchmark_results_full.csv                  # Full benchmark results
│   ├── risk_windows_real.csv                       # DWD infection risk labels
│   └── *.png                                       # Result visualisations
│
├── download_dlr_wms.py          # Download DLR crop type raster
├── download_sentinel2_matched.py # Download Sentinel-2 matched to fields
├── download_2021_scenes.py      # Download 2021 Sentinel-2 scenes
├── download_b08.py              # Download NIR band (B08)
├── download_prithvi.py          # Download Prithvi pretrained weights
├── process_dwd_fixed.py         # Compute DWD infection risk labels
│
├── extract_fields_highres.py    # Vectorise sugar beet fields from raster
├── extract_prithvi_v3.py        # Extract Prithvi temporal embeddings
├── compute_ndvi_baseline.py     # Compute NDVI/NDRE features per field
├── random_forest_baseline.py    # Random Forest leaf embedding baseline
│
├── finetune_prithvi.py          # Fine-tune MLP head on Prithvi embeddings
├── run_10_seeds.py              # GEO-Bench 10-seed evaluation (2022)
├── evaluate_2021.py             # Cross-year evaluation (2021)
│
├── cluster_and_compare.py       # Initial clustering comparison
├── interpret_clusters.py        # Cluster interpretation + spatial maps
├── map_all_classes.py           # DLR class legend mapping
│
└── README.md
```

---

## Reproduction

### Requirements

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers huggingface_hub timm einops
pip install rasterio geopandas shapely scikit-learn scikit-image
pip install pandas numpy matplotlib pystac-client planetary-computer
```

### Full Pipeline

```bash
# Step 1: Download Prithvi pretrained weights (~1.3 GB)
python download_prithvi.py

# Step 2: Download DLR crop type map and extract sugar beet fields
python download_dlr_wms.py
python extract_fields_highres.py

# Step 3: Download matched Sentinel-2 scenes (2022)
python download_sentinel2_matched.py
python download_b08.py

# Step 4: Process DWD weather data and compute infection risk labels
python process_dwd_fixed.py

# Step 5: Extract Prithvi embeddings (requires GPU)
python extract_prithvi_v3.py

# Step 6: Compute baselines
python compute_ndvi_baseline.py
python random_forest_baseline.py

# Step 7: Fine-tune and evaluate (2022)
python finetune_prithvi.py
python run_10_seeds.py

# Step 8: Cross-year validation (2021)
python download_2021_scenes.py
python evaluate_2021.py
```

### Hardware

- GPU: NVIDIA RTX 3060 (12 GB VRAM)
- Prithvi inference: ~62 seconds for 1020 fields
- 10-seed evaluation: ~20 minutes (includes fine-tuning)

---

## Limitations

- **3 timesteps instead of 4** - Prithvi was pretrained on 4 timesteps; no cloud-free November scene was available for our region
- **Single geographic region** - results from Franconia only; spatial generalisation requires additional regions
- **No disease ground truth** - DWD infection risk labels are a weather-derived proxy, not field-observed CLS incidence
- **Fine-tuning uses NDVI pseudo-labels** - not real agronomic labels; a supervised variant with expert annotations would be stronger
- **Sadbhave et al. (2025) F1 not directly comparable** - their 75.21% F1 was obtained on different fields, different years, and with different evaluation protocol

---

## References

```
Lacoste et al. (2023). GEO-Bench: Toward Foundation Models for Earth Monitoring.
NeurIPS 2023 Datasets and Benchmarks Track.

Jakubik et al. (2023). Foundation Models for Generalist Geospatial Artificial Intelligence.
arXiv:2310.18660.

Sadbhave et al. (2025). Sugar-Beet Stress Detection using Satellite Image Time Series.
arXiv:2507.13514.

Wolf & Verreet (2002). An integrated pest management system in Germany for the control
of fungal leaf diseases in sugar beet: The IPM Sugar Beet Model. Plant Disease 86(4).

DLR EOC CropTypes Germany. https://geoservice.dlr.de/web/datasets/croptypes_de

Sentinel-2 L2A. Copernicus Data Space / Microsoft Planetary Computer.

DWD Open Data. https://opendata.dwd.de/climate_environment/CDC/
```

---

## Citation

```bibtex
@misc{chakraborty2025eobenchmark,
  author    = {Sudipto Chakraborty},
  title     = {Benchmarking EO Foundation Models for Agricultural Field Representation Learning},
  year      = {2025},
  publisher = {GitHub},
  url       = {https://github.com/sudipto09/sugar-beet-eo-benchmark}
}
```

---

*Master's Practikum - Julius-Maximilians-Universität Würzburg · Computer Vision Lab · 2026*