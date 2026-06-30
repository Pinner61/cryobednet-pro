.PHONY: install data train eval figures app test clean

install:
	python -m pip install --upgrade pip
	pip install -e .

data:
	python scripts/make_mock_data.py --out data/mock_tiles.npz --n 640 --seed 42

train:
	python -m cryobednet.train --config configs/mock_cpu.yaml

eval:
	python -m cryobednet.evaluate --config configs/mock_cpu.yaml --split holdout

figures:
	python scripts/export_figures.py --run outputs/mock_cpu --split holdout

app:
	streamlit run app/streamlit_app.py

test:
	pytest -q

clean:
	rm -rf outputs mlruns logs .pytest_cache
