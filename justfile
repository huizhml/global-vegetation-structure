# justfile — single entrypoint for every Hydra run=<name> in the repo.
#
# Run from the repo root:  just <recipe> [extra hydra overrides...]
# List everything grouped:  just            (or: just --list)
#
# Design (Option B): every recipe is TOP-LEVEL (so zsh tab-completion of
# recipe names always works) and PACKAGE-PREFIXED so completion scopes:
#     just viz-<TAB>    visualization recipes
#     just dl-<TAB>     download recipes
#     just post-<TAB>   postprocessing recipes
#     just eval-<TAB>   evaluation recipes
#     just tools-<TAB>  tools recipes
# `[group(...)]` only affects how `just --list` buckets them.
#
# `*args` is a Hydra passthrough — append any override, e.g.
#     just viz-datacube run.bg_color=black run.rh_step=4
#     just viz-global-mosaic-pdf run.cmap=cividis run.cmax=30
#
# Requires just >= 1.27 (the [group(...)] attribute). conda-forge build is fine.

viz   := "python -m visualization.run"
dl    := "python -m download.run"
post  := "python -m postprocessing.run"
evalp := "python -m evaluation.run"
tools := "python -m tools.run"
preproc := "python -m preprocessing.run"

# Show all available recipes, grouped
default:
    @just --list

# ─────────────────────────────────────────────────────────────────────────
# visualization  (python -m visualization.run)
# ─────────────────────────────────────────────────────────────────────────

# Resample blended tiles and build a global mosaic (Hydra's default run)
[group('visualization')]
viz-resample-and-mosaic *args:
    {{viz}} run=resample_and_mosaic {{args}}

# Global mosaic from the STAC catalog
[group('visualization')]
viz-global-mosaic *args:
    {{viz}} run=create_global_mosaic {{args}}

# Global prediction-interval (diff) mosaic
[group('visualization')]
viz-global-diff-mosaic *args:
    {{viz}} run=create_global_diff_mosaic {{args}}

# Global relative prediction-interval mosaic
[group('visualization')]
viz-global-relative-diff-mosaic *args:
    {{viz}} run=create_global_relative_diff_mosaic {{args}}

# Global mosaic PDF, one page per band (generic — pass run.product, run.cmap, ...)
[group('visualization')]
viz-global-mosaic-pdf *args:
    {{viz}} run=create_global_mosaic_pdf {{args}}

# Global mosaic PDF for RH98 prediction intervals (the documented example)
[group('visualization')]
viz-global-mosaic-pdf-rh98 *args:
    {{viz}} run=create_global_mosaic_pdf \
        run.product=prediction_intervals \
        run.tif_filename_pattern='*RH98*.tif' \
        run.cmap=mako run.cmax=30 run.cmin=0 {{args}}

# 3-D datacube render (override run.bg_color=black for dark background)
[group('visualization')]
viz-datacube *args:
    {{viz}} run=visualize_datacube {{args}}

# Check a mosaic after bias correction
[group('visualization')]
viz-check-after-bias-correction *args:
    {{viz}} run=check_after_bias_correction {{args}}

# RH-pair diagnostic PDF (issue tiles)
[group('visualization')]
viz-pdf-thumb *args:
    {{viz}} run=create_pdf_thumb {{args}}

# Prediction vs 8-neighbor diagnostic PDF
[group('visualization')]
viz-pred-neighbor-pdf *args:
    {{viz}} run=create_pred_neighbor_pdf {{args}}

# Cloud-cover boxplot PDF
[group('visualization')]
viz-cloud-cover-boxplot *args:
    {{viz}} run=create_cloud_cover_boxplot {{args}}

# ─────────────────────────────────────────────────────────────────────────
# download  (python -m download.run)
# ─────────────────────────────────────────────────────────────────────────

# Download SOTA canopy-height maps
[group('download')]
dl-sota-chms *args:
    {{dl}} run=download_sota_chms {{args}}

# Download ForestTemp data
[group('download')]
dl-forest-temp *args:
    {{dl}} run=download_forest_temp {{args}}

# Build the MGRS grid
[group('download')]
dl-get-mgrs *args:
    {{dl}} run=get_mgrs {{args}}

# Add GEDI count to the MGRS grid
[group('download')]
dl-add-gedi-count *args:
    {{dl}} run=add_gedi_count {{args}}

# Download all valid GEDI points
[group('download')]
dl-all-valid-gedi *args:
    {{dl}} run=download_all_valid_gedi {{args}}

# Add slope to GEDI points
[group('download')]
dl-add-slope *args:
    {{dl}} run=add_slope {{args}}

# Check slope distribution of GEDI points
[group('download')]
dl-check-slope-distribution *args:
    {{dl}} run=check_slope_distribution {{args}}

# Get the GEDI points actually used
[group('download')]
dl-get-used-gedi-points *args:
    {{dl}} run=get_used_gedi_points {{args}}

# Get tiles covered by GEDI
[group('download')]
dl-get-tiles-covered-by-gedi *args:
    {{dl}} run=get_tiles_covered_by_gedi {{args}}

# Gather Sentinel-2 metadata
[group('download')]
dl-s2-meta-gather *args:
    {{dl}} run=s2_meta_gather {{args}}

# Sentinel-2 best candidates (API)
[group('download')]
dl-s2-best-candidates-api *args:
    {{dl}} run=s2_best_candidates_api {{args}}

# Sentinel-2 best candidates
[group('download')]
dl-s2-best-candidates *args:
    {{dl}} run=s2_best_candidates {{args}}

# Sentinel-2 download
[group('download')]
dl-s2-download *args:
    {{dl}} run=s2_download {{args}}

# Aggregate growing season
[group('download')]
dl-agg-growing-season *args:
    {{dl}} run=agg_growing_season {{args}}

# Download inference inputs
[group('download')]
dl-inference *args:
    {{dl}} run=download_inference {{args}}

# Download downstream-task data
[group('download')]
dl-downstream-task-data *args:
    {{dl}} run=download_downstream_task_data {{args}}

# ─────────────────────────────────────────────────────────────────────────
# postprocessing  (python -m postprocessing.run)
# ─────────────────────────────────────────────────────────────────────────

# Mask snow/water predictions
[group('postprocessing')]
post-mask-snow-water-preds *args:
    {{post}} run=mask_snow_water_preds {{args}}

# Update STAC collection
[group('postprocessing')]
post-update-stac-collection *args:
    {{post}} run=update_stac_collection {{args}}

# Create updated STAC collection
[group('postprocessing')]
post-create-updated-stac-collection *args:
    {{post}} run=create_updated_stac_collection {{args}}

# Make parquet subcolumns
[group('postprocessing')]
post-make-parq-subcolumns *args:
    {{post}} run=make_parq_subcolumns {{args}}

# Get tiles to reblend
[group('postprocessing')]
post-get-tiles-reblend *args:
    {{post}} run=get_tiles_reblend {{args}}

# Repartition data
[group('postprocessing')]
post-repartition-data *args:
    {{post}} run=repartition_data {{args}}

# Extract GEDI from H5
[group('postprocessing')]
post-extract-gedi-from-h5 *args:
    {{post}} run=extract_gedi_from_h5 {{args}}

# Extract predictions
[group('postprocessing')]
post-extract-pred *args:
    {{post}} run=extract_pred {{args}}

# Sample VSM patches
[group('postprocessing')]
post-sample-vsm-patches *args:
    {{post}} run=sample_vsm_patches {{args}}

# Check after bias correction
[group('postprocessing')]
post-check-after-bias-correction *args:
    {{post}} run=check_after_bias_correction {{args}}

# Check two partitioned datasets
[group('postprocessing')]
post-check-two-partitioned-datasets *args:
    {{post}} run=check_two_partitioned_datasets {{args}}

# Check two datasets
[group('postprocessing')]
post-check-two-datasets *args:
    {{post}} run=check_two_datasets {{args}}

# Make manifest
[group('postprocessing')]
post-make-manifest *args:
    {{post}} run=make_manifest {{args}}

# Add ours to SOTA GEDI
[group('postprocessing')]
post-add-ours-to-sota-gedi *args:
    {{post}} run=add_ours_to_sota_gedi {{args}}

# Add ours (blended) to SOTA GEDI
[group('postprocessing')]
post-add-ours-blended-to-sota-gedi *args:
    {{post}} run=add_ours_blended_to_sota_gedi {{args}}

# Evaluate bias correction
[group('postprocessing')]
post-evaluate-bias-correction *args:
    {{post}} run=evaluate_bias_correction {{args}}

# Get tiles without enough GEDI ground truth
[group('postprocessing')]
post-get-tiles-wo-enough-gedi-gt *args:
    {{post}} run=get_tiles_wo_enough_gedi_gt {{args}}

# Run blending
[group('postprocessing')]
post-run-blending *args:
    {{post}} run=run_blending {{args}}

# Add biome
[group('postprocessing')]
post-add-biome *args:
    {{post}} run=add_biome {{args}}

# Create distance maps
[group('postprocessing')]
post-create-distance-maps *args:
    {{post}} run=create_distance_maps {{args}}

# Create VRT
[group('postprocessing')]
post-create-vrt *args:
    {{post}} run=create_vrt {{args}}

# Translate predictions
[group('postprocessing')]
post-translate-predictions *args:
    {{post}} run=translate_predictions {{args}}

# Get redundant tiles
[group('postprocessing')]
post-get-tiles-redundant *args:
    {{post}} run=get_tiles_redundant {{args}}

# Get nodata tiles
[group('postprocessing')]
post-get-tiles-nodata *args:
    {{post}} run=get_tiles_nodata {{args}}

# Sample forest temp
[group('postprocessing')]
post-sample-forest-temp *args:
    {{post}} run=sample_forest_temp {{args}}

# ─────────────────────────────────────────────────────────────────────────
# evaluation  (python -m evaluation.run)
# ─────────────────────────────────────────────────────────────────────────

# Evaluate VSM on GEDI
[group('evaluation')]
eval-vsm-on-gedi *args:
    {{evalp}} run=evaluate_vsm_on_gedi {{args}}

# Evaluate CHM with SOTA
[group('evaluation')]
eval-chm-with-sota *args:
    {{evalp}} run=evaluate_chm_with_sota {{args}}

# Evaluate CHM with ALS and LVIS
[group('evaluation')]
eval-chm-with-als-and-lvis *args:
    {{evalp}} run=evaluate_chm_with_als_and_lvis {{args}}

# Generate diversity-indices map
[group('evaluation')]
eval-generate-diversity-indices-map *args:
    {{evalp}} run=generate_diversity_indices_map {{args}}

# Sample points by biome
[group('evaluation')]
eval-sample-points-by-biome *args:
    {{evalp}} run=sample_points_by_biome {{args}}

# Partition points by tile
[group('evaluation')]
eval-partition-points-by-tile *args:
    {{evalp}} run=partition_points_by_tile {{args}}

# Compute entropy
[group('evaluation')]
eval-compute-entropy *args:
    {{evalp}} run=compute_entropy {{args}}

# Compute diversity indices
[group('evaluation')]
eval-compute-diversity-indices *args:
    {{evalp}} run=compute_diversity_indices {{args}}

# Evaluate diversity indices
[group('evaluation')]
eval-evaluate-diversity-indices *args:
    {{evalp}} run=evaluate_diversity_indices {{args}}

# Plot biome combined boxplot
[group('evaluation')]
eval-plot-biome-combined-boxplot *args:
    {{evalp}} run=plot_biome_combined_boxplot {{args}}

# Plot residuals RH98 binned
[group('evaluation')]
eval-plot-residuals-rh98-bined *args:
    {{evalp}} run=plot_residuals_rh98_bined {{args}}

# Calc S2 patch stats
[group('evaluation')]
eval-cal-s2-patch-stats *args:
    {{evalp}} run=cal_s2_patch_stats {{args}}

# Calc alpha-EM patch stats
[group('evaluation')]
eval-cal-alpha-em-patch-stats *args:
    {{evalp}} run=cal_alpha_em_patch_stats {{args}}

# Calc VSM-17 patch stats
[group('evaluation')]
eval-cal-vsm-17-patch-stats *args:
    {{evalp}} run=cal_vsm_17_patch_stats {{args}}

# Calc VSM patch stats
[group('evaluation')]
eval-cal-vsm-patch-stats *args:
    {{evalp}} run=cal_vsm_patch_stats {{args}}

# Merge patch stats
[group('evaluation')]
eval-merge-patch-stats *args:
    {{evalp}} run=merge_patch_stats {{args}}

# Run naturalness classification
[group('evaluation')]
eval-run-naturalness-classification *args:
    {{evalp}} run=run_naturalness_classification {{args}}

# Plot bars (spatial context)
[group('evaluation')]
eval-plot-bars-spatial-context *args:
    {{evalp}} run=plot_bars_spatial_context {{args}}

# Plot bars (center pixel)
[group('evaluation')]
eval-plot-bars-center-pixel *args:
    {{evalp}} run=plot_bars_center_pixel {{args}}

# Compute GLCM texture
[group('evaluation')]
eval-compute-glcm-texture *args:
    {{evalp}} run=compute_glcm_texture {{args}}

# ─────────────────────────────────────────────────────────────────────────
# tools  (python -m tools.run)
# ─────────────────────────────────────────────────────────────────────────

# Update the README
[group('tools')]
tools-update-readme *args:
    {{tools}} run=update_readme {{args}}

# ─────────────────────────────────────────────────────────────────────────
# misc entrypoints (different invocation shape — generic passthrough)
# ─────────────────────────────────────────────────────────────────────────

# Model training / eval — LightningCLI: pass a subcommand (fit|validate|test|predict)
#   just train fit --config config/....yaml
[group('misc')]
train *args:
    python run.py {{args}}

# preprocessing entrypoint (single hydra config, no registered run= names)
[group('misc')]
pre *args:
    {{preproc}} {{args}}
