# Data Sources

CryoBedNet is structured for multi-source Antarctic raster inputs. The mock data generator is only for software testing.

## Recommended real datasets

| Layer | Role | Notes |
|---|---|---|
| BEDMAP2 / BedMachine Antarctica bed | coarse bed prior | use as low-resolution input |
| REMA surface elevation | high-resolution auxiliary predictor | useful for surface morphology |
| MEaSUREs ice velocity | high-resolution auxiliary predictor | links terrain and ice dynamics |
| snow accumulation / SMB | auxiliary predictor | climate and mass balance context |
| grounded/floating mask | auxiliary predictor | helps avoid invalid ocean/non-ice cells |
| Operation IceBridge / radar products | target/evaluation | local high-resolution bed constraints |

## Preprocessing rules

1. Reproject all rasters to Antarctic Polar Stereographic, commonly EPSG:3031.
2. Align pixel grids exactly before tiling.
3. Create a valid-data mask before sampling tiles.
4. Keep train, validation, and test regions spatially separated.
5. Store only data pointers in Git. Use DVC or external storage for heavy rasters.

## Final tensor format

```text
x_lr   [N, 1, H, W]
a_hr   [N, C, H*scale, W*scale]
y_hr   [N, 1, H*scale, W*scale]
region [N]
tile_id[N]
```
