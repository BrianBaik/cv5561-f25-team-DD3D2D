# cv5561-f25-team-DD3D2D

Project Title: Diffusion-Driven 3D Reconstruction from 2D Images

Short desctiption: Using Difusion Models to build 3D reconstructions given 2D images.

Members: Sungmin Baik (baik0025@umn.edu), Parees Pradhan (pradh086@umn.edu), Adil Arya (arya0033@umn.edu)

Roles: 
  Common tasks - 3D Geometry & Rendering, Demonstration, Documentation & Writing
  Sungmin Baik - Systems & Infrastructure
  Parees Pradhan - Data & Evaluation 
  Adil Arya - Model & Training


The main things you can do with this repo are:

- Train a DDPM model on a dataset of images  
- Monitor training loss and visualize convergence  
- Generate denoised samples from random noise or corrupted images  
- View and compare saved results and graphs  

---

## 1. Environment Setup

### 1.1. Python & virtual environment

```bash
# create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate      # on macOS / Linux
# .venv\Scripts\activate       # on Windows (PowerShell or cmd)

# install dependencies
pip install -r requirements.txt
```

If you don’t have a `requirements.txt`, install at least:

```bash
pip install torch torchvision matplotlib numpy
```

---

## 2. Data Preparation

1. Place your training images in a folder, for example:
   - `data/train/` – for training images  
   - `data/val/`   – for validation images (optional)

2. The dataset loader in the code expects images to be:
   - RGB or grayscale
   - All the same size (or will be resized inside the dataset class)
   - Common formats like `.png` / `.jpg`

> **Note:** If your dataset path or image size is hard-coded in `train_ddpm.py` or a config section, update it there and make sure it matches what you write in this README.

---

## 3. Training the Diffusion Model

The main training entry point is:

```bash
python train_ddpm.py
```

Typical useful arguments (if your script supports them):

```bash
python train_ddpm.py \
  --data_dir data/train \
  --epochs 100 \
  --batch_size 64 \
  --lr 1e-4 \
  --img_size 64 \
  --save_dir outputs/
```

Where:

- `--data_dir` : path to training images  
- `--epochs`   : number of training epochs  
- `--batch_size`: batch size  
- `--lr`       : learning rate  
- `--img_size` : image resolution used by the model  
- `--save_dir` : base folder where checkpoints, samples, and plots are saved  

### 3.1. What happens during training

During training, `train_ddpm.py` will:

- Load the dataset and create dataloaders  
- Initialize the U-Net / DDPM model and optimizer  
- Loop over timesteps and noise levels according to the beta schedule  
- Compute the diffusion loss (usually MSE on predicted noise)  
- **Log loss per iteration and per epoch** (printed to terminal and saved to a log file / CSV)  
- Periodically save:
  - Model checkpoints  
  - Sample denoised images  
  - Training loss curve plot  

---

## 4. Getting Results

### 4.1. Saved checkpoints

By default, trained model checkpoints are saved under:

```text
outputs/checkpoints/
```

Files typically look like:

```text
outputs/checkpoints/ddpm_epoch_XX.pt
```

Where `XX` is the epoch number.

You can later load one of these checkpoints for sampling or further training.

---

### 4.2. Generated images / samples

During or after training, the script saves generated images to:

```text
outputs/samples/
```

Common patterns:

- `outputs/samples/epoch_XX_samples.png` – a grid of denoised images generated at the end of epoch `XX`
- `outputs/samples/final_samples.png`     – samples from the final model

If your code supports reconstructing from corrupted inputs, you may also see:

- `outputs/samples/epoch_XX_noisy.png`      – noisy inputs  
- `outputs/samples/epoch_XX_denoised.png`   – corresponding denoised outputs  

---

### 4.3. Loss logs & training graphs

Training statistics are saved in:

```text
outputs/logs/train_loss.csv      # raw numeric log (epoch, loss)
outputs/figures/loss_curve.png   # plotted loss vs. epoch
```

- `train_loss.csv` can be opened in Excel or plotted manually.  
- `loss_curve.png` shows how the training loss decreases over epochs.

If your script only prints loss to the terminal, you can still use the per-epoch losses shown in the console to describe training behavior in your report. (If you changed filenames or directories in `train_ddpm.py`, update the paths above to match.)

---

## 5. Reproducing the Main Results

To reproduce the main qualitative and quantitative results:

1. **Train the model**

   ```bash
   python train_ddpm.py \
     --data_dir data/train \
     --epochs 100 \
     --batch_size 64 \
     --lr 1e-4 \
     --save_dir outputs/
   ```

2. **Check training convergence**

   - Open `outputs/figures/loss_curve.png` to verify the loss is decreasing.  
   - Optionally inspect `outputs/logs/train_loss.csv` to see exact values.

3. **Inspect generated images**

   - Look at `outputs/samples/epoch_XX_samples.png` across epochs to see visual improvement.  
   - Use `outputs/samples/final_samples.png` (or equivalent) as a final qualitative result.

4. **Evaluation script**

   If you have an evaluation or sampling script like:

   ```bash
   python eval_ddpm.py --checkpoint outputs/checkpoints/ddpm_epoch_100.pt
   ```

   It will typically:
   - Load the specified checkpoint  
   - Generate a batch of samples  
   - Save them under `outputs/samples/`  
   - Possibly compute evaluation metrics (PSNR, SSIM, FID, etc.) and log them under `outputs/logs/`.


