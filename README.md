# Information-Preserving Adaptive Candidate Selection for Underwater Image Enhancement

This repository contains the code, checkpoints, local demos, dataset viewer, analysis artifacts, and reported evaluation for a paired underwater image enhancement study. The research explores generating several controlled enhancement candidates and selecting one based on the input and candidate characteristics.

The central idea is **generate candidates, then select**. The experiments investigate whether candidate selection can balance fidelity, information preservation, and reconstruction-based consequence. Results show candidate variation, but the V7 selector favors Aggressive and does not outperform V4 on every metric. Read the limitations below before describing V7 as adaptive or superior.

## At a glance

- Dataset: paired EUVP Underwater ImageNet subset, expected locally under `data/EUVP/.../underwater_imagenet/`.
- Dataset split: 2,960 train, 370 validation, and 370 test pairs (80/10/10, seed 42), defined in `data/splits/` and created by `src/data/create_splits.py`.
- Main frozen evaluation checkpoint: `checkpoints/v7/paper1_v7_best_epoch_019.pth` (epoch 19).
- Evaluation report: `results/v7/v7_final_test_report.md` and its JSON companion.
- Demo ports: V3 7860, V4 7861, V7 7862. Dataset viewer 8765.
- The EUVP dataset is excluded by `.gitignore`; obtain it separately and place it at the expected path.
- There is no root `requirements.txt` or Colab requirements file in this checkout. Install dependencies explicitly as described below.

## Research progression

V3 through V7 are successive experimental versions, not unrelated models:

| Version | Implementation / research role |
|---|---|
| V3 | Preservation-oriented baseline. Demo: `demo/app.py`; model: `src/models/paper1_model.py`. |
| V4 | Candidate-specific Conservative, Balanced, and Aggressive generation. Demo: `demo/app_v4.py`; model: `src/models/paper1_model_v4.py`. |
| V5 | Multi-objective candidate selection; explores fidelity, preservation, and consequence objectives. |
| V6 | Fidelity-constrained selection experiment. |
| V7 | Condition-aware utility selection over the V4 candidate space. Demo: `demo/app_v7.py`; model: `src/models/paper1_model_v7.py`. |

Related training scripts are `src/train_v3.py` through `src/train_v7.py`; losses are in `src/losses/`. Earlier versions V2 and the base model also have code and checkpoints. Consult each training report and diagnostic before comparing versions.

## V7 inference and reference images

V7 inference requires only an underwater input image. In `Paper1ModelV7.forward(x)`, the model encodes the input condition, generates the three candidates, calculates preservation scores and consequence errors, computes a utility for each candidate, applies softmax, and selects the maximum-utility candidate (argmax). A reference image is not part of this forward path. The demos accept an optional aligned reference for PSNR/SSIM evaluation; it is not needed to enhance an image.

V7 reuses the V4 candidate space and frozen condition, generator, preservation, and consequence components; its selector is condition-aware. Its scorer combines candidate features, condition features, their elementwise interaction, preservation score, and consequence error. See `src/models/paper1_model_v7.py` for the authoritative implementation.

## Frozen checkpoint and reported results

The final frozen held-out V7 evaluation in this checkout uses:

```text
checkpoints/v7/paper1_v7_best_epoch_019.pth
Epoch 19; SHA-256 A0E04FEEC9A6C69CCEE0DD8A6E23E9C35227CDAA10C270DD82F853536920B49E
```

The evaluation contains 370 test pairs. Reported selected-output metrics are:

| Metric | V7 selected |
|---|---:|
| PSNR | 21.2027 dB |
| SSIM | 0.789558 |
| UIQM | 1.6703 |
| UCIQE | 5.6111 |
| Preservation score | 0.965145 |
| Consequence error | 0.081368 |

Test selection counts: Conservative 51 (13.78%), Balanced 62 (16.76%), Aggressive 257 (69.46%). V4 selected Aggressive on all 370 images in the reported comparison and achieved higher PSNR (23.3363) and SSIM (0.817886) than V7. V7's results therefore show selection variation, with a strong Aggressive bias; they do not establish that it is optimal or superior for every input. UIQM/UCIQE are no-reference quality measures and should not be interpreted as paired-reference accuracy.

These values are transcribed from `results/v7/v7_final_test_report.md`. The report states the test split was used only for this final frozen evaluation, not training, tuning, or checkpoint selection. V7 training configuration and validation-only findings are in `results/v7/v7_training_report.md`. Keep reported artifacts and the checkpoint unchanged when reproducing these results. Conduct new experiments in separately named outputs and do not tune against the held-out test set.

## Repository map

| Path | Purpose |
|---|---|
| `src/models/` | V3–V7 model modules and component implementations. |
| `src/losses/` | Version-specific training losses. |
| `src/data/` | EUVP paired dataset, transforms, and split creation. |
| `src/train_v*.py` | Versioned training entry points. Training requires the dataset and writes checkpoints/results. |
| `src/evaluate_v7_final_test.py` | Frozen V7 test evaluation entry point. |
| `demo/` | V3, V4, and V7 Gradio demos. |
| `checkpoints/` | Versioned model checkpoints, including the frozen V7 checkpoint. |
| `results/v7/` | V7 training and final test reports. |
| `results/` | Versioned evaluations, inference examples, and analysis artifacts. |
| `diagnostics/` | Selector and candidate-space analyses. |
| `tools/dataset_viewer/` | Local read-only paired dataset browser and its own README. |
| `tools/figures/` | Generated figure artifacts and metadata. |
| `docs/` | Research notes, including preservation-method documentation. |
| `PROJECT_AUDIT.md` | Additional project audit notes. |

The manuscript named in the supplied project description (`ICSIE26_Sundari_MSEC.docx`) is not present in this checkout. Add or link to the approved manuscript separately if it should be part of the Git handoff.

## Requirements and setup

Use Python 3.10+ (the original draft's Python 3.12 environment is also suitable). The repository does not currently provide a dependency lockfile or requirements file. The demos and model code import PyTorch, Gradio, NumPy, and Pillow; data/evaluation utilities also use packages such as scikit-image and tqdm. Install compatible versions for your Python and GPU environment. For example, install a PyTorch build appropriate to the machine first, then:

```powershell
python -m pip install gradio numpy pillow scikit-image tqdm torchvision
```

Create a Windows virtual environment from the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell activation is restricted, invoke `.\.venv\Scripts\python.exe` directly. Check PyTorch and CUDA availability with:

```powershell
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

The demo can run on CPU, though more slowly. Original training used seed 42, batch size 8, Adam at 1e-4, up to 20 epochs, and selected checkpoints using validation loss. See the per-version training script/report for exact settings. Do not assume the current machine matches the historical GPU environment.

## Dataset setup

Download/obtain the paired EUVP Underwater ImageNet data separately. The expected layout is:

```text
data/EUVP/EUVP-Dataset/EUVP/Paired/underwater_imagenet/
├── trainA/   # degraded input images
└── trainB/   # paired reference images
```

The dataset is intentionally ignored by Git. The paired loader and split/evaluation code verify paired names and use the split manifests under `data/splits/`. Split generation is available as:

```powershell
python src/data/create_splits.py
```

Only regenerate splits when intentionally starting a new experiment: doing so can change the manifests used by the frozen evaluation. The evaluation report describes its test preprocessing as RGB tensors in [0,1], with no resizing or random augmentation. Interactive demos resize uploaded images to 256 × 256; this is a separate demo preprocessing path.

## Run the demos

Run each command from the repository root. Keep the terminal open and visit the corresponding local URL.

### V3

```powershell
python demo/app.py
```

Open <http://127.0.0.1:7860>. Uses `checkpoints/v3/paper1_v3_best.pth`.

### V4

```powershell
python demo/app_v4.py
```

Open <http://127.0.0.1:7861>. Uses the V4 checkpoint configured in the demo.

### V7

```powershell
python demo/app_v7.py
```

Open <http://127.0.0.1:7862>. Uses `checkpoints/v7/paper1_v7_best_epoch_019.pth`. It displays the candidates, selector information, and selected result; optional reference input enables reference-based metrics. The V7 app also includes a Figure 5 builder/export workflow.

## Dataset viewer

The local, read-only viewer browses matching `trainA`/`trainB` image pairs and helps select three representative samples. It supports filename search, random browsing, pair selection, heuristic representative suggestions, figure PNG export, and filename copying. Suggestions are simple image-statistic heuristics and should be reviewed by a person. The viewer uses the originals directly and does not modify the dataset.

From the repository root:

```powershell
python tools/dataset_viewer/server.py --host 127.0.0.1 --port 8765
```

Open <http://127.0.0.1:8765>. The viewer has no additional Python package dependencies. It requires the dataset at the path above.

## Google Colab

Colab can run inference without a local GPU laptop. Clone the repository, select **Runtime → Change runtime type → GPU**, and install the same Python dependencies in Colab's Linux environment. Do not copy a Windows `.venv`.

```python
!git clone https://github.com/HariniS1207/Paper1_UIE.git
%cd Paper1_UIE
!pip install gradio numpy pillow scikit-image tqdm torchvision
```

Install a compatible CUDA-enabled PyTorch version if the runtime does not already have one. Verify it:

```python
import torch
print(torch.__version__, torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
```

The repository checkout includes the checkpoint, so an EUVP download is not needed for single-image V7 inference. Upload an image in Colab, then run:

```python
import sys
from pathlib import Path
import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

root = Path('/content/Paper1_UIE')
sys.path.insert(0, str(root / 'src'))
from models.paper1_model_v7 import Paper1ModelV7

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load(root / 'checkpoints/v7/paper1_v7_best_epoch_019.pth',
                        map_location='cpu', weights_only=False)
model = Paper1ModelV7().to(device)
model.load_state_dict(checkpoint['model_state_dict'], strict=True)
model.eval()

# Set image_path to a Colab uploaded image. Evaluation preserves input dimensions;
# the web demo instead resizes images to 256 x 256.
image_path = '/content/underwater.jpg'
image = Image.open(image_path).convert('RGB')
x = (pil_to_tensor(image).float() / 255.0).unsqueeze(0).to(device)
with torch.inference_mode():
    output = model(x)
selected = output['selected_image'][0].clamp(0, 1).cpu()
result = Image.fromarray((selected.permute(1, 2, 0).numpy() * 255).round().astype('uint8'))
display(result)
print('Selected:', model.candidate_names[int(output['selected_index'][0])])
```

For an interactive Colab demo, note that the checked-in apps bind to `127.0.0.1` and local ports; those addresses are not automatically exposed in Colab. Direct model inference in the notebook is the documented workflow here. Colab runtime GPU type and installed package versions can vary.

## Reproducing evaluations

The V7 final test evaluation command is:

```powershell
python src/evaluate_v7_final_test.py
```

It requires the paired dataset and split manifests. Review the script before running: it evaluates the held-out set and writes evaluation artifacts. The recorded evaluation used 370 pairs and `checkpoints/v7/paper1_v7_best_epoch_019.pth`. Do not use the test set to select checkpoints, tune settings, or choose qualitative examples.

Training entry points include `src/train_v3.py`, `src/train_v4.py`, `src/train_v5.py`, `src/train_v6.py`, and `src/train_v7.py`. Training is not needed to use the demos or run single-image inference. Training requires the dataset and writes checkpoints/results; preserve the frozen artifacts by making any new run clearly separate.

## Figures and paper relationship

`tools/figures/` contains generated figure files and metadata; `results/` contains outputs and reports. The dataset viewer is a sample-selection utility, not a paper figure. Demo screenshots are interface views, not substitutes for model result figures. Use the metadata/report files to identify the checkpoint and source images associated with generated examples. The manuscript itself is not included in the current checkout.

## What is frozen

For the reported V7 test result, treat the epoch 19 checkpoint, test split/manifests, preprocessing, evaluation report, and associated figure metadata as frozen evidence. Do not overwrite them or silently change the reported metrics. New training, checkpoint selection, or test protocols should be labeled as new experiments and saved separately. The code remains available for inspection and new research; the frozen designation refers to the artifacts underlying the current report.

## Quick commands

```powershell
python demo/app.py                         # V3: http://127.0.0.1:7860
python demo/app_v4.py                      # V4: http://127.0.0.1:7861
python demo/app_v7.py                      # V7: http://127.0.0.1:7862
python tools/dataset_viewer/server.py      # Viewer: http://127.0.0.1:8765
```
