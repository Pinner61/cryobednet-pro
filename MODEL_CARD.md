# CryoBedNet Model Card

## Model

CryoBedNet predicts high-resolution Antarctic bed-elevation tiles from a low-resolution bed prior and high-resolution auxiliary channels.

## Intended use

- Research portfolio demonstration
- Geospatial super-resolution experiments
- Cryosphere ML workflow prototyping
- Visualization and model-comparison exercises

## Not intended for

- Operational sea-level forecasts
- Scientific claims without real-data validation
- Replacing expert glaciology workflows

## Training data

The current repository ships with a mock data generator. Real experiments should use curated Antarctic rasters with documented preprocessing, masking, and region-separated evaluation.

## Evaluation

Primary metrics are MAE, RMSE, PSNR, SSIM, and slope RMSE. The required baseline is bicubic interpolation. The required split is region holdout.

## Risks

Super-resolution models can hallucinate terrain texture. Results should be validated against independent radar/raster products and interpreted as model outputs, not direct observations.
