# AI-Based Restoration of Degraded Images for Semiconductor Inspection

KLA Problem Statement — Hackathon 2026 (SEMICON India)

Two-stage restoration pipeline: **DnCNN** (denoising) → **ESPCN** (2× super-resolution),
recovering clean, full-resolution wafer inspection images from inputs degraded by
speckle noise, additive Gaussian noise, and downsampling.

A joint (single-network) model is the stretch-goal upgrade once this baseline is
validated — see [Project Roadmap](#roadmap) below.

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── train.py                  # trains DnCNN and/or ESPCN
├── inference.py               # standalone inference: input-dir -> output-dir
├── configs/
│   └── default.yaml           # hyperparameters, paths, degradation params
├── src/
│   ├── data/
│   │   ├── degradation.py     # speckle + Gaussian + downsample, randomized order
│   │   └── dataset.py         # synthetic-pair and official-pair PyTorch datasets
│   ├── models/
│   │   ├── dncnn.py
│   │   └── espcn.py
│   ├── pipeline.py            # TwoStageRestorer: chains denoiser + SR at inference
│   └── utils/
│       └── metrics.py         # PSNR, SSIM, LPIPS
├── weights/                   # trained checkpoints (.pth) — not committed, see below
├── results/                   # sample restored outputs + metric summaries
├── webapp/                    # interactive Flask demo (upload -> restore -> view)
│   ├── app.py
│   └── templates/index.html
└── solution_presentation.pptx
```

## Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Tested with Python 3.10+, PyTorch 2.1+. A CUDA-capable GPU is strongly
recommended for training; inference falls back to CPU automatically if no
GPU is available (`--device cpu`).

## Data Setup

**Option A — synthetic degradation (default, no external data needed to start):**

Download a general-purpose image dataset (DIV2K or BSD500) and point
`configs/default.yaml` at it:

```yaml
data:
  gt_dir: "data/gt_train"
  val_gt_dir: "data/gt_val"
```

Degraded (NoisyLR) pairs are generated on the fly from these clean images
using `src/data/degradation.py` — exact ground truth is guaranteed since we
control the degradation.

**Option B — official KLA dataset:**

Download the official paired GT/NoisyLR dataset from the hackathon portal,
preserve its folder structure, then set:

```yaml
data:
  use_official: true
  official_gt_dir: "data/official/GT"
  official_noisy_dir: "data/official/NoisyLR"
```

## Training

```bash
# Train both stages (denoiser, then super-resolver on denoiser's output)
python train.py --config configs/default.yaml --stage both

# Or train stages independently for debugging/benchmarking
python train.py --config configs/default.yaml --stage denoise
python train.py --config configs/default.yaml --stage sr
```

Checkpoints are written to `weights/`:
- `dncnn_best.pth` / `dncnn_final.pth`
- `espcn_best.pth` / `espcn_final.pth`

Random seed, config, and per-epoch validation PSNR are logged to stdout for
reproducibility; redirect to a file if you want a persistent training log
(`python train.py ... | tee logs/train_$(date +%s).log`).

## Inference

Standalone script, no source edits required — only CLI arguments:

```bash
python inference.py \
  --input-dir  path/to/NoisyLR \
  --output-dir path/to/restored \
  --weights-dir weights \
  --batch-size 16 \
  --device cuda
```

- Loads every image in `--input-dir`, restores it, writes it to
  `--output-dir` with the same filename.
- Batched GPU execution (`--batch-size`); falls back to CPU if no GPU is
  available or `--device cpu` is passed.
- Prints end-to-end runtime (disk I/O + preprocessing + transfer + model +
  postprocessing + save), matching KLA's runtime definition.

### Input/Output contract

- **Input**: any of `.png .jpg .jpeg .bmp .tif .tiff` in `--input-dir`. Not
  clipped or renormalized on load — the model handles values outside
  [0,1], since KLA's NoisyLR data may legitimately extend beyond that
  range.
- **Output**: same filename, written to `--output-dir`, clipped to [0,1]
  and saved as 8-bit. Clipping is applied exactly once, at save time — per
  KLA's note that they score the image "exactly as saved by the submitted
  pipeline."
- **Batching assumption**: images within a single batch are assumed to be
  the same resolution (as KLA's spec states: ~256×256 or ~512×512). Mixed
  sizes within one batch are handled by zero-padding to the batch's max
  size and cropping back per-image after inference, but for best fidelity
  keep same-resolution images in the same batch (or use `--batch-size 1`
  for a fully mixed-size folder).

## Interactive Web Demo

A small Flask app for showing the pipeline in action in a browser — useful
for the hackathon's required prototype/demo video.

```bash
pip install -r requirements.txt   # includes flask + pillow
python webapp/app.py
```

Then open **http://localhost:5000**. Drag in an image and click "Restore
Image." Check "synthetically degrade it first" if you're uploading a clean
image and want to see the full pipeline (degrade → restore) rather than
supplying an already-degraded sample.

This reuses `src/pipeline.py` directly — it's the same restoration code as
`inference.py`, just wrapped in a browser UI, not a separate mock. It
auto-loads `weights/dncnn_best.pth` and `weights/espcn_best.pth` if
present; if you haven't trained yet, it runs in a clearly-labeled **demo
mode** with a randomly initialized model rather than pretending to show
real results.

## Evaluation

`src/utils/metrics.py` implements PSNR, SSIM, and LPIPS (KLA's three
required metrics). LPIPS needs `pip install lpips` (already in
`requirements.txt`) and downloads a small pretrained backbone on first
use.

Validation-split PSNR is logged automatically during training (see
`train.py`). A full metrics report against a held-out validation set,
plus the mandatory baseline comparison, restored-image examples (success
+ failure cases), and runtime/hardware details, lives in
`results/` and `solution_presentation.pptx` — fill these in once training
completes.

## Roadmap

See `solution_presentation.pptx` for the full write-up. Short version:
build and validate the two-stage baseline first (fast, debuggable,
diagnosable), then attempt a joint U-Net/Real-ESRGAN-style model as a
stretch goal, benchmarked directly against this baseline on the same test
set.

## Known Limitations / Honest Caveats

- Synthetic speckle + Gaussian noise + downsampling approximates real
  inspection-hardware degradation; results may not transfer perfectly to
  real scans.
- No real paired degraded/clean inspection images are available, so the
  model can only be validated quantitatively on synthetic data and
  qualitatively on real wafer-style images (no ground truth).
- The two-stage pipeline can propagate stage-1 (denoising) errors into
  stage 2 (super-resolution); the joint-model stretch goal exists
  specifically to address this.

## External Resources

- DIV2K / BSD500 — general-purpose SR training data
- WM-811K — public wafer-map dataset, used for qualitative real-domain
  checks (no ground truth available)
- DnCNN (Zhang et al., 2017), SRCNN (Dong et al., 2015), ESPCN (Shi et
  al., 2016), Real-ESRGAN (Wang et al., 2021) — architectural references
