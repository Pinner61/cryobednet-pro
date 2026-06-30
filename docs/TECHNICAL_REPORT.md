# CryoBedNet Technical Report

## Research question

Can a neural super-resolution model reconstruct sharper Antarctic bed-elevation structure from a coarse bed prior and aligned high-resolution auxiliary cryosphere layers?

## Modeling approach

CryoBedNet frames Antarctic bed reconstruction as supervised image-to-image super-resolution. The model receives a coarse bed-elevation tile and high-resolution auxiliary channels, then predicts a high-resolution bed-elevation tile.

## Model families

1. **Residual U-Net SR**: stable baseline architecture with skip connections and residual prediction over bicubic upsampling.
2. **RRDB generator**: compact ESRGAN-inspired model using residual dense blocks.
3. **Patch discriminator**: optional adversarial component for sharper terrain features.

## Losses

The main training objective is L1 reconstruction loss plus a terrain-gradient loss. The gradient term penalizes incorrect slope and ridge/valley structure, which matters for bed-topography realism.

## Evaluation

Metrics:

- MAE
- RMSE
- PSNR
- SSIM
- slope RMSE

The main comparison is against bicubic interpolation. The most important split is a region holdout, where one glacier region is excluded from training.

## Sustainability relevance

Higher-quality bed topography can support more realistic glacier and ice-sheet modeling. Better subglacial terrain information can reduce uncertainty in ice flow simulations and long-term sea-level risk assessment.

## Limitations

This repository currently runs end-to-end on mock terrain tiles. Real scientific conclusions require curated Antarctic datasets, careful masking, independent validation, and expert review.
