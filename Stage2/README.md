# Stage 2: Video Anomaly Detection

Stage 2 uses pre-extracted CLIP visual features to train the Temporal Adapter (TeA) and Visual Prompt Fusion (VPF) modules of VPTeA-VAD. The CLIP image and text encoders remain frozen, while video-level labels are used for coarse-grained detection and fine-grained anomaly localization.

## Prerequisites

Please follow the [official VadCLIP repository](https://github.com/nwpu-zxr/VadCLIP/tree/main) for dataset download, video splits, CLIP feature extraction, and ground-truth preparation. The Stage 2 training scripts read `.npy` features and CSV lists rather than raw videos.

The following resources are also required:

1. A PyTorch, NumPy, Pandas, and CUDA environment.
2. CLIP ViT-B/16 weights. `src/model.py` loads `ViT-B/16` through the local CLIP implementation. The weights may need to be downloaded or placed in the local CLIP cache before the first run.
3. The UCF-Crime and XD-Violence CLIP `.npy` features, CSV lists, ground-truth files, and corresponding knowledge bases prepared following VadCLIP. Run from the Stage 2 directory so that the default relative paths resolve correctly. The feature paths in the CSV files must be valid, and the feature dimension must match the default `--visual-width 512`.

## UCF-Crime Training and Inference

Pretrained UCF-Crime model weights are provided through Baidu Netdisk:

[Download the pretrained model](https://pan.baidu.com/s/1O1ShtqWh6QCyrhHSWp2J0A?pwd=dqr7) (extraction code: `dqr7`)

After downloading the model, place it at the default path `model/model_ucf.pth`. After completing the data and CLIP preparation described above, you can directly evaluate the pretrained model without retraining.

Run UCF inference from the Stage 2 directory:

```powershell
cd Stage2
python src\ucf_test.py
```

The script uses the provided `model/model_ucf.pth` by default and prints the detection performance, including AUC, AP, and average mAP. To train or retrain the UCF model, run:

```powershell
python src\ucf_train.py
```

UCF uses the following defaults:

| Configuration | Default |
| --- | --- |
| Training list | `list/ucf_CLIP_rgb.csv` |
| Test list | `list/ucf_CLIP_rgbtest.csv` |
| Knowledge base | `src/ucf_knowledge_base.txt` |
| Ground truth | `list/gt_ucf.npy` |
| Temporal segments | `list/gt_segment_ucf.npy` |
| Class labels | `list/gt_label_ucf.npy` |
| Pretrained model | `model/model_ucf.pth` |
| Batch size | `32` |
| Maximum epochs | `5` |
| Learning rate | `5e-6` |

## XD-Violence Training and Inference

XD follows the same workflow as UCF. Run training from the Stage 2 directory:

```powershell
python src\xd_train.py
```

XD uses `list/xd_CLIP_rgb.csv`, `list/xd_CLIP_rgbtest.csv`, `list/gt.npy`, `list/gt_segment.npy`, `list/gt_label.npy`, and `src/xd_knowledge_base.txt` by default. The default batch size is 96, the maximum number of epochs is 15, the model is saved to `model/model_xd.pth`, and the checkpoint is saved to `model/checkpoint_xd.pth`.

After training generates `model/model_xd.pth`, run XD inference:

```powershell
python src\xd_test.py
```

## Knowledge Bases

This project provides the knowledge-base contents used in the official experiments:

- `src/ucf_knowledge_base.txt`
- `src/xd_knowledge_base.txt`

UCF training and inference use `src/ucf_knowledge_base.txt` by default. XD training and inference use `src/xd_knowledge_base.txt` by default. Use the corresponding file directly unless the category definitions have been changed.

Although the files use the `.txt` extension, both files must contain JSON objects that can be parsed by `json.load`. The UCF knowledge base must contain `positive` and `negative` prompts for the 13 anomalous classes. The XD knowledge base must contain the corresponding prompts for `B1`, `B2`, `B4`, `B5`, `B6`, and `G`. Do not add `Normal` or the XD `A` class as an anomaly prompt.

## Overriding Parameters

Training parameters are defined in `src/ucf_option.py` and `src/xd_option.py` and can also be overridden from the command line, for example:

```powershell
python src\ucf_train.py --batch-size 16 --max-epoch 5
```

The training scripts create the `model/` directory automatically. Before enabling checkpoint loading, make sure that the corresponding checkpoint exists and that the configuration matches the selected dataset.

