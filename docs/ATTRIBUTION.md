# Attribution

CryoBedNet is a new implementation inspired by DeepBedMap:

- W. J. Leong and H. J. Horgan, **DeepBedMap: a deep neural network for resolving the bed topography of Antarctica**, The Cryosphere, 2020.
- Original repository: `weiji14/deepbedmap`.

The file `assets/reference/deepbedmap_architecture_reference.pdf` and the license copy in `assets/reference/DEEPBEDMAP_LICENSE.md` come from the DeepBedMap repository. They are included as reference material only.

This repository does not claim ownership of the original DeepBedMap paper, figures, datasets, or code. The new code in `src/cryobednet/` is a separate PyTorch implementation created for educational and research-portfolio use.

Recommended public wording:

> CryoBedNet is inspired by DeepBedMap and reimplements the core idea in PyTorch with region holdout evaluation, SOL/HPC training, MLflow-ready experiment tracking, and an interactive dashboard.
