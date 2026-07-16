# Benchmarking EO Foundation Models for Agricultural Field Representation Learning

**Master's Practikum — M.Sc. Aerospace Informatics**
**Julius-Maximilians-Universität Würzburg · Computer Vision Lab**
**Advisor:** Dr. Fayaz Ali Dharejo · Prof. Dr. Radu Timofte

---

## Research Question

Do temporal embeddings from EO foundation models (Prithvi EO 2.0) produce better 
representations of sugar beet field behaviour than classical spectral baselines — 
and does domain-specific fine-tuning improve this?

---

## Methodology

Evaluation protocol follows **GEO-Bench** (Lacoste et al., NeurIPS 2023):
- 10 random seeds
- Interquartile Mean (IQM) as primary metric
- 95% bootstrapped confidence intervals
- Normalized aggregated scores

Task definition extends **Sadbhave et al. (2025)** — sugar beet temporal 
stress detection from Sentinel-2 time series.

---

## Models Benchmarked

| Model | Type |
|---|---|
| Prithvi EO 2.0 300M (zero-shot) | EO Foundation Model — temporal |
| Prithvi EO 2.0 (fine-tuned head) | EO Foundation Model — adapted |
| NDVI/NDRE baseline | Spectral index — temporal |
| Random Forest (leaf embeddings) | Classical ML |
| Raw Spectral KMeans | Classical baseline |

---

## Dataset

- **Fields:** 1020 real sugar beet field polygons
- **Source:** DLR EOC CropTypes Germany (class 13, 10m resolution)
- **Region:** Franconia, Bavaria, Germany
- **Imagery:** Sentinel-2 L2A (Copernicus, via Microsoft Planetary Computer)
- **Years:** 2021, 2022 (3 scenes per year: Apr/Jun/Aug)
- **Weather:** DWD Station 05705 (Würzburg), hourly 2020–2022
- **All data sources are public — no proprietary data used**

---

## Results (2022, n=1020, 10 seeds, IQM)

| Model | Silhouette ↑ | 95% CI | Davies-Bouldin ↓ | 95% CI |
|---|---|---|---|---|
| Prithvi (fine-tuned) | **0.343** | [0.330, 0.363] | **0.832** | [0.798, 0.876] |
| Prithvi (zero-shot) | 0.336 | [0.336, 0.336] | 0.893 | [0.893, 0.893] |
| NDVI baseline | 0.256 | [0.256, 0.256] | 1.113 | [1.113, 1.113] |
| Raw Spectral KMeans | 0.179 | [0.179, 0.179] | 1.489 | [1.488, 1.489] |

---

## Key Findings

1. **Prithvi outperforms spectral baselines geometrically** — higher silhouette
   and lower Davies-Bouldin in 2022 under GEO-Bench evaluation protocol.

2. **Fine-tuning improves both metrics** — a lightweight MLP head trained on
   NDVI pseudo-labels improves Prithvi geometry (Sil: 0.336→0.343, DB: 0.893→0.832).

3. **Zero-shot Prithvi clusters are not agronomically interpretable** — all 5
   clusters show nearly identical NDVI trajectories, suggesting the model captures
   low-level spectral texture rather than crop phenology without domain adaptation.

4. **Cross-year ranking is not fully stable** — in 2021, NDVI baseline outperforms
   Prithvi zero-shot, partly attributed to anomalous April NDVI values in the
   2021 scenes. Foundation model embeddings show greater robustness to this.

---

## Reproduction

### Environment
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers huggingface_hub timm einops rasterio geopandas
pip install scikit-learn scikit-image pandas numpy matplotlib pystac-client planetary-computer
```

### Download Prithvi weights
```bash
python download_prithvi.py
```

### Run pipeline
```bash
# 1. Download data
python download_dlr_wms.py
python download_sentinel2_matched.py
python download_b08.py

# 2. Extract fields
python extract_fields_highres.py

# 3. Extract Prithvi embeddings
python extract_prithvi_v3.py

# 4. Compute baselines
python compute_ndvi_baseline.py
python random_forest_baseline.py

# 5. Fine-tune and evaluate
python finetune_prithvi.py
python run_10_seeds.py

# 6. Cross-year validation
python download_2021_scenes.py
python evaluate_2021.py
```

---

## References

- Lacoste et al. (2023). GEO-Bench: Toward Foundation Models for Earth Monitoring. NeurIPS.
- Jakubik et al. (2023). Foundation Models for Generalist Geospatial Artificial Intelligence. arXiv.
- Sadbhave et al. (2025). Sugar-Beet Stress Detection using Satellite Image Time Series. arXiv:2507.13514.
- Wolf & Verreet (2002). An integrated pest management system in Germany for the control of fungal leaf diseases in sugar beet. Plant Disease.
