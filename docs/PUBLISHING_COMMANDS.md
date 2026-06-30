# Build, run, and publish commands

## 1. Create project locally

```bash
mkdir cryobednet-pro
unzip cryobednet_pro.zip -d cryobednet-pro
cd cryobednet-pro
```

## 2. Create environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .[dash,dev]
```

For Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .[dash,dev]
```

## 3. Run smoke experiment

```bash
python scripts/make_mock_data.py --out data/mock_tiles.npz --n 160 --seed 42
python -m cryobednet.train --config configs/mock_cpu.yaml
python -m cryobednet.evaluate --config configs/mock_cpu.yaml --split holdout
python scripts/export_figures.py --run outputs/mock_cpu --split holdout
```

## 4. Open dashboard

```bash
streamlit run app/streamlit_app.py
```

Use:

```text
outputs/mock_cpu/predictions_holdout.npz
```

## 5. Run tests

```bash
pytest -q
```

## 6. Git setup

```bash
git init
git add README.md pyproject.toml environment.yml requirements.txt Makefile dvc.yaml Dockerfile LICENSE CITATION.cff MODEL_CARD.md .gitignore .github app assets configs docs reports scripts slurm src tests
git commit -m "Initial CryoBedNet implementation"
```

## 7. Push to GitHub

Create an empty GitHub repo named `cryobednet-pro`, then run:

```bash
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/cryobednet-pro.git
git push -u origin main
```

## 8. Add results without pushing heavy files

```bash
mkdir -p reports/figures
cp outputs/mock_cpu/loss_curve.png reports/figures/
cp outputs/mock_cpu/prediction_panel_holdout.png reports/figures/
cp outputs/mock_cpu/transect_holdout.png reports/figures/
cp outputs/mock_cpu/error_hist_holdout.png reports/figures/
git add reports/figures README.md
git commit -m "Add initial experiment figures"
git push
```

Do not commit `.npz`, `.pt`, `.tif`, `.nc`, or SOL logs.
