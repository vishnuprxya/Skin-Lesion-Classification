# Explainable Hybrid CNN-Transformer for Skin Lesion Classification

## 1a. LOW-DATA option (recommended if you can't download 5-6 GB)
Same dataset, but download only two small files (no image folders):
```
pip install kaggle
mkdir -p data/ham10000
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000 -f HAM10000_metadata.csv -p data/ham10000 --unzip
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000 -f hmnist_28_28_RGB.csv  -p data/ham10000 --unzip
python train.py --data_dir data/ham10000 --pixels_csv hmnist_28_28_RGB.csv --side 28 --img_size 128 --epochs 30
```
(Note: this dataset version only ships 28x28 and 8x8 pixel files; use the 28x28 RGB one.) You can also download these two files by hand from the dataset's *Data* tab on Kaggle.

## 1b. Full images (6 GB)
Kaggle dataset: **Skin Cancer MNIST: HAM10000** (`kmader/skin-cancer-mnist-ham10000`)
```
pip install kaggle
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000 -p data/ham10000 --unzip
```
Folder must contain `HAM10000_metadata.csv` and the two image folders.

## 2. Install & train
```
pip install -r requirements.txt
python train.py --data_dir data/ham10000 --epochs 30 --out_dir outputs
```
(GPU strongly recommended; on Colab/Kaggle T4 ~3-4 min/epoch.)

## 3. Dashboard
```
streamlit run app.py
```
Files: `model.py` (CNN + Transformer blocks + attention), `explain.py` (Grad-CAM, rollout), `train.py`, `app.py`.
