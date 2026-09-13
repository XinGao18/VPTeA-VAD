# Stage 1: Prompt Pair Knowledge base Generation

Stage 1 uses a vision-language model to build and iteratively update the anomaly detection knowledge base through verbalized learning. The Learner predicts video-level anomaly labels, while the Optimizer updates positive and negative prompt pairs using video-level labels and visual evidence, providing clearer anomaly boundaries for Stage 2.

## Prerequisites

Please follow the [official VERA repository](https://github.com/vera-framework/VERA) for Stage 1 environment setup, dataset preparation, and frame extraction. This repository does not provide a separate `requirements.txt`; use the VERA environment configuration as the dependency reference.

The following resources are also required:

1. A CUDA-capable GPU. `training.py` loads the vision-language model on the GPU with `bfloat16`.
2. InternVL model weights. The script loads the model from `pretrained/InternVL3-8B` by default. If the model is stored elsewhere, update `path` in `training.py`.
3. Video frames and the corresponding JSON annotations. Stage 1 samples images from the frame data rather than reading raw videos directly; video identifiers and frame lengths must match the annotations.
4. The provided `xd_train.json`, `xd_val.json`, `UCF_Instruct_train.json`, `UCF_Instruct_val.json`, and `knowledge_base_seeds.json` files. `knowledge_base_seeds.json` initializes category semantics and is required. See [Stage1/Data/README.md](Data/README.md) for additional information.

## Default Training: XD-Violence

Run the command from the Stage 1 directory so that the relative paths `Data/...` and `pretrained/...` resolve correctly:

```powershell
cd Stage1
python training.py
```

Before running, open `training.py` and change the default XD frame root to the local dataset path:

```python
xd_frame_root = 'G:/xd/'
```

Set `xd_frame_root` to the local root directory of XD-Violence.

The default annotation files are:

```python
train_ann_root = 'Data/xd_train.json'
val_ann_root = 'Data/xd_val.json'
```

The current default configuration samples 8 frames per video, uses a batch size of 2, and trains for 5 epochs. Accuracy is recorded during validation and the knowledge base is updated during training.

## Switching to UCF-Crime

`training.py` already contains the UCF configuration, but it is commented out by default. To switch datasets:

1. Uncomment the UCF dataset block using `Data/UCF_Instruct_train.json` and `Data/UCF_Instruct_val.json`.
2. Set the UCF `vis_root` to the local UCF dataset root.
3. Comment out the active XD configuration, including `xd_frame_root`, the XD training and validation annotations, and the corresponding `train_dataset` and `val_dataset` initialization.
4. Keep `knowledge_base = _knowledge_base_for_dataset(train_dataset, train_ann_root)` unchanged. The script selects the UCF knowledge base from the annotations.
5. Run the following command from the Stage 1 directory:

```powershell
python training.py
```

The UCF annotation files are:

```python
train_ann_root = 'Data/UCF_Instruct_train.json'
val_ann_root = 'Data/UCF_Instruct_val.json'
```

The XD and UCF configurations must remain mutually exclusive so that variables from both datasets are not defined in the same run.

## Output Files

The main outputs are generated in or relative to the Stage 1 directory:

- `Data/XD_knowledge_base.json` or `Data/UCF_knowledge_base.json`: JSON knowledge base state that can be loaded and updated across runs.
- `stage1_generated_kb.txt`: knowledge-base content and validation accuracy recorded after validation.
- `log_generating_KB.txt`: Learner, Optimizer, knowledge-base update, and parsing-failure logs.
- `logs_VPTeA/`: TensorBoard log directory.

If a knowledge-base state file already exists, the script loads it and continues updating it. Stage 2 uses `Stage2/src/xd_knowledge_base.txt` and `Stage2/src/ucf_knowledge_base.txt`; when connecting a newly generated knowledge base, make sure its JSON category set matches the Stage 2 dataset.
