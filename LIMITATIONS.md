# Limitations

## Labels and targets
- DLR CropTypes is a classified (random-forest) product, not field-surveyed
  ground truth. Some polygons will be misclassified or mixed.
- There are no disease or stress labels. All targets are NDVI-derived
  trajectory quantities (green-up, senescence, amplitude, held-out date), so
  the study measures what the model encodes about vegetation dynamics, not
  crop-health detection.
- Because targets are computed from the same Sentinel-2 bands the models see,
  some probe targets are partly reconstructable from the inputs. Layer-wise
  results flag these (`construction_ceiling` column); interpret them as an upper
  bound, not as evidence of learned phenology.

## Data
- Only three dates per season (Apr, Jun, Aug). Prithvi EO 2.0 supports four
  timesteps, so the temporal context is shorter than the model's design.
- One region (around Würzburg) and one crop. Results may not transfer.
- 2021 and 2022 are a season contrast (cooler/wetter vs hot/dry), not replicates,
  and acquisition dates differ by 5-10 days between years.
- Polygon sources differ by year: 2022 from the DLR WMS, 2021 cut from the V021
  GeoTIFF (class 60 = sugar beet; the WMS numbering differs). Field sets are
  therefore not identical across years.
- Sentinel-2 processing baseline changes the radiometric offset: baseline
  >= 04.00 adds +1000 DN (2022 scenes), earlier baselines do not (2021 scenes).
  This is handled via `scene_metadata.json` and checked with
  `check_processing_baseline.py`; any new scene must be verified the same way.
- Prithvi was pretrained on 30 m HLS imagery; native 10-20 m input is
  out of distribution. The 30 m variant reduces but does not remove this gap.

## Method
- Probes are linear (ridge) on frozen features; a nonlinear probe could change
  the ranking. Gradient-boosting probes are reported for the held-out task but
  not for all experiments.
- 2021 held-out R^2 is near zero for every feature set, so comparisons in that
  year have little power.
- Insensitivity to date order (CKA >= 0.99 under shuffle/reverse) is consistent
  with temporal pooling but does not prove it; the three-date input and field
  averaging may also contribute.
- Confidence intervals are field-level bootstraps; they do not capture spatial
  autocorrelation between neighbouring fields.
- Deterministic models (frozen features, NDVI baselines) have no seed variance,
  so seed-based intervals are degenerate for them; bootstrap over fields is used
  instead.

## Scope
- SatMAE and DINOv2 are planned but not yet evaluated.
- Fine-tuning is out of scope. Earlier fine-tuned clustering numbers came from
  decoder output rather than encoder features and must not be reported.