import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import decord
from decord import VideoReader, cpu
import random
import torch
from torch.utils.data.dataloader import default_collate
from PIL import Image
from typing import Dict, Optional, Sequence
import transformers
import json
import re
import pickle
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
import copy
import math
from torchvision import transforms
import pdb
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel
import pytorch_lightning as pl
import itertools

from datasets_qwen.video_instruction_dataset import Video_Instruct_Dataset
import math
import torch
from transformers import AutoTokenizer, AutoModel
from knowledge_base import (
    KnowledgeBaseError,
    LearnerDecision,
    load_or_create_knowledge_base,
    parse_optimizer_update,
    render_optimizer_prompt,
)

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

_KB_LOG_PATH = './log_generating_KB.txt'


def _response_snippet(response, limit=500):
    return ' '.join(str(response).split())[:limit]


def _write_kb_log(event, payload):
    record = dict(payload)
    record['event'] = event
    with open(_KB_LOG_PATH, 'a', encoding='utf-8') as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n')


def _sample_case(target, prediction):
    return ('TP' if prediction == 1 else 'FN') if target == 1 else (
        'FP' if prediction == 1 else 'TN'
    )


def _parse_training_decisions(responses, knowledge_base):
    decisions = []
    errors = []
    predictions = []
    for response in responses:
        try:
            decision = knowledge_base.parse_learner_decision(response)
        except KnowledgeBaseError as exc:
            fallback_prediction = _binary_prediction(response)
            decisions.append(LearnerDecision(fallback_prediction))
            errors.append(str(exc))
            predictions.append(fallback_prediction)
        else:
            decisions.append(decision)
            errors.append(None)
            predictions.append(decision.prediction)
    return decisions, errors, predictions


def _sample_records(
    video_names, targets, true_label_sets, decisions, errors, predictions, responses,
):
    records = []
    for name, target, true_labels, decision, error, prediction, response in zip(
        video_names, targets, true_label_sets, decisions, errors, predictions, responses
    ):
        records.append({
            'video_name': str(name),
            'prediction': prediction,
            'target': target,
            'true': sorted(true_labels),
            'matched': decision.matched_labels if decision else [],
            'related': decision.related_labels if decision else [],
            'case': _sample_case(target, prediction),
            'attribution_valid': error is None,
            'attribution_error': error,
            'raw_response': _response_snippet(response),
        })
    return records



def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    print(world_size, 'xxx')
    num_layers = {
        'InternVL2-1B': 24, 'InternVL2-2B': 24, 'InternVL2-4B': 32, 'InternVL2-8B': 32,
        'InternVL2-26B': 48, 'InternVL2-40B': 60, 'InternVL2-Llama3-76B': 80}[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map


def VPTeA_collate(batch):
    """Collate tensor fields while preserving per-sample class-label sets."""
    fields = list(zip(*batch))
    result = []
    for values in fields:
        first = values[0]
        if isinstance(first, torch.Tensor):
            result.append(default_collate(values))
        elif isinstance(first, (int, np.integer)):
            result.append(torch.tensor(values, dtype=torch.long))
        elif isinstance(first, set):
            result.append([set(value) for value in values])
        elif isinstance(first, list) and first and isinstance(first[0], str):
            result.append([list(items) for items in zip(*values)])
        else:
            result.append(list(values))
    return tuple(result)


def _sample_label_sets(cls_labels, batch_size):
    if cls_labels is None:
        return [set() for _ in range(batch_size)]
    if isinstance(cls_labels, set):
        cls_labels = [cls_labels]
    values = list(cls_labels)
    if len(values) != batch_size:
        values = values[:batch_size] + [set()] * max(0, batch_size - len(values))
    return [set(value) if value else set() for value in values]


def _binary_prediction(response):
    output = str(response).rsplit('Output', 1)[-1]
    labeled = re.search(r'prediction\s*["\']?\s*:\s*([01])', output, re.IGNORECASE)
    if labeled is not None:
        return int(labeled.group(1))
    match = re.search(r'(?<!\d)([01])(?!\d)', output)
    if match is not None:
        return int(match.group(1))
    return 0 if '0' in output and '1' not in output else 1


def _knowledge_base_for_dataset(dataset, annotation_path):
    return load_or_create_knowledge_base(dataset.annotation, annotation_path)

class VideoAnomalyDetectionModel(pl.LightningModule):
    def __init__(self, model, tokenizer, optimizer_instruct, model_instruct, generation_config, knowledge_base, epochs=5):
        super().__init__()
        self.model = model
        self.processor = tokenizer
        self.optimizer_instruct = optimizer_instruct
        self.model_instruct = model_instruct
        self.kb = knowledge_base
        self.generation_config = generation_config
        self.epochs = epochs
        self.validation_step_outputs = []
        self.validation_step_count = []
        self.automatic_optimization = False

    def forward(self, frame_path, batch_idx=0):
        self.model.eval()
        batch_size = len(frame_path[0])
        messages = []
        for i in range(batch_size):
            frame_path_list = [frame_path[j][i] for j in range(len(frame_path))]
            messages.append([{
                "role": "user",
                "content": [{"type": "video", "video": frame_path_list, "fps": 1.0}],
            }])

        sys_start = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n'
        sys_end = '<|im_end|>\n<|im_start|>assistant\n'
        video_prefix = '<|vision_start|><|video_pad|><|vision_end|>'
        question = self.model_instruct.replace('[$Data]', video_prefix)
        rendered_kb = json.dumps(self.kb.render_all(), ensure_ascii=False, indent=2)
        question = question.replace('[$KnowledgeBase]', rendered_kb)
        question = sys_start + question + sys_end

        questions = [question] * batch_size
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=questions, images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to("cuda")
        generated_ids = self.model.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        responses = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return responses, messages

    def training_step(self, batch, batch_idx):
        self.model.eval()
        if len(batch) == 4:
            frame_path, labels, video_name, cls_labels = batch
        elif len(batch) == 3:
            frame_path, labels, video_name = batch
            cls_labels = None
        else:
            raise ValueError(f"unexpected Qwen batch with {len(batch)} fields")
        labels = torch.as_tensor(labels, dtype=torch.long, device="cuda")
        true_label_sets = _sample_label_sets(cls_labels, len(labels))

        responses, messages = self.forward(frame_path, batch_idx)
        decisions, attribution_errors, predict_labels = _parse_training_decisions(
            responses, self.kb
        )
        sample_records = _sample_records(
            video_name, labels.detach().cpu().tolist(), true_label_sets,
            decisions, attribution_errors, predict_labels, responses,
        )
        _write_kb_log('KB_SAMPLE', {'batch_idx': batch_idx, 'samples': sample_records})

        scope = self.kb.build_update_scope(
            labels.detach().cpu().tolist(), true_label_sets, decisions
        )
        parse_failures = sum(error is not None for error in attribution_errors)
        _write_kb_log('KB_SCOPE', {
            'batch_idx': batch_idx, 'scope': scope,
            'status': 'partial' if parse_failures and scope else ('skipped' if not scope else 'ready'),
            'parse_failures': parse_failures,
        })

        if not scope:
            _write_kb_log('KB_DECISION', {
                'batch_idx': batch_idx, 'action': 'skipped',
                'reason': 'empty update scope',
            })
            _write_kb_log('KB_UPDATE', {
                'batch_idx': batch_idx, 'status': 'skipped',
                'changes': [], 'reason': 'empty update scope',
            })
        else:
            optimizer_prompt = render_optimizer_prompt(
                self.optimizer_instruct,
                '<|vision_start|><|video_pad|><|vision_end|>',
                sample_records, scope, self.kb,
            )
            batch_frame_path = [
                frame_path[j][i]
                for i in range(len(frame_path[0]))
                for j in range(len(frame_path))
            ]
            opt_message = [{
                "role": "user",
                "content": [{"type": "video", "video": batch_frame_path, "fps": 1.0}],
            }]
            image_inputs, video_inputs = process_vision_info(opt_message)
            opt_inputs = self.processor(
                text=optimizer_prompt, images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt",
            ).to("cuda")
            generated_ids = self.model.generate(**opt_inputs, max_new_tokens=1024)
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(opt_inputs.input_ids, generated_ids)
            ]
            response = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            try:
                parsed = parse_optimizer_update(
                    response, scope=scope, allowed_labels=self.kb.labels
                )
                result = self.kb.apply_optimizer_update(
                    parsed, scope=scope, allowed_labels=self.kb.labels, save=True
                )
            except KnowledgeBaseError as exc:
                _write_kb_log('KB_DECISION', {
                    'batch_idx': batch_idx, 'action': 'rejected',
                    'reason': str(exc), 'response': _response_snippet(response),
                })
                _write_kb_log('KB_UPDATE', {
                    'batch_idx': batch_idx,
                    'status': 'rejected',
                    'reason': str(exc),
                    'response': _response_snippet(response),
                })
            else:
                _write_kb_log('KB_DECISION', {
                    'batch_idx': batch_idx, 'action': result.action,
                    'response': _response_snippet(response),
                })
                _write_kb_log('KB_UPDATE', {
                    'batch_idx': batch_idx,
                    'status': 'applied',
                    'changes': [change.as_dict() for change in result.changes],
                })

        predict_labels = torch.tensor(predict_labels, device=labels.device)
        correct_predictions = (predict_labels == labels).sum()
        accuracy = correct_predictions / len(labels)
        self.log(
            'train_acc', accuracy.item(), prog_bar=True, on_step=True,
            sync_dist=True, batch_size=len(labels),
        )
        return torch.tensor(1.0, requires_grad=True, device=labels.device) 

     
    def validation_step(self, batch, batch_idx):
        if len(batch) == 4:
            frame_path, labels, video_name, _ = batch
        elif len(batch) == 3:
            frame_path, labels, video_name = batch
        else:
            raise ValueError(f"unexpected Qwen batch with {len(batch)} fields")
        labels = torch.as_tensor(labels, dtype=torch.long, device="cuda")
        responses, messages = self.forward(frame_path, batch_idx)
        predict_labels = torch.tensor(
            [_binary_prediction(response) for response in responses], device=labels.device
        )
        correct_predictions = (predict_labels == labels).sum()
        accuracy = correct_predictions / len(labels)
        self.log('val_acc', accuracy.item(), prog_bar=True, on_step=True, sync_dist=True, batch_size=len(labels))
        self.validation_step_outputs.append(correct_predictions)
        self.validation_step_count.append(len(labels))
        return {'val_acc': accuracy}
    

    def on_validation_epoch_end(self):
        # Calculate average accuracy across all batches in the validation set
        epoch_average = torch.stack(self.validation_step_outputs).sum()/sum(self.validation_step_count)
        self.log("val_avg_acc", epoch_average.item(), sync_dist=True)
        self.validation_step_outputs.clear()  # free memory
        self.validation_step_count.clear()  # free memory

        f=open(text_file, 'a')
        f.write(f'accuracy: {epoch_average.item()}'+'\n')
        f.write('Knowledge base:\n')
        f.write(json.dumps(self.kb.render_all(), ensure_ascii=False) + '\n')
        f.write('\n')
        f.close()
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

text_file = './vml_generated_question_qwen.txt'

f = open(text_file, 'w')
f.writelines('Generative Questions\n')
f.close()

 
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
).eval()


# default processer
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")


# Create the datasets for UCF-Crime.
# train_ann_root = 'Data/UCF_Instruct_train.json'
# train_dataset = Video_Instruct_Dataset(
#     vis_root='Data/ucf/', ann_root=train_ann_root, num_sampled_frame=8
# )
# val_dataset = Video_Instruct_Dataset(
#     vis_root='Data/ucf/', ann_root='Data/UCF_Instruct_val.json',
#     TEST_FLAG=True, num_sampled_frame=8
# )

# Create the datasets for XD-Violence.
# The Qwen dataset loader derives the sampled frame paths from this root.
xd_frame_root = 'G:/xd/'
train_ann_root = 'Data/xd_train.json'
val_ann_root = 'Data/xd_val.json'
train_dataset = Video_Instruct_Dataset(
    vis_root=xd_frame_root, ann_root=train_ann_root
)
val_dataset = Video_Instruct_Dataset(
    vis_root=xd_frame_root, ann_root=val_ann_root, TEST_FLAG=True
)

# Create the data loaders
knowledge_base = _knowledge_base_for_dataset(train_dataset, train_ann_root)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=16, drop_last=False, collate_fn=VPTeA_collate)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=1, drop_last=False, collate_fn=VPTeA_collate)


 
file = open('VPTeA_learner_instruct.txt','r')
model_instruct =  file.read()
file.close()

file = open('VPTeA_optimizer_instruct.txt','r')
optimizer_instruct =  file.read()
file.close()


# Initialize Lightning Model
lightning_model = VideoAnomalyDetectionModel(
    model=model, tokenizer=processor, optimizer_instruct=optimizer_instruct,
    model_instruct=model_instruct, generation_config=None,
    knowledge_base=knowledge_base,
)

from pytorch_lightning.loggers import TensorBoardLogger

tb_logger = TensorBoardLogger(save_dir="logs_qwen_debug_finish/")

# Train the model using the Lightning Trainer
trainer = pl.Trainer(
    logger=tb_logger,
    val_check_interval=100,
    log_every_n_steps=5,
    max_epochs=50, 
    enable_checkpointing=False,
    devices=1,  # Use all available GPUs
    accelerator="gpu",  # GPU training
    strategy="ddp"  # Distributed Data Parallel training
)


TRAIN_FLAG=True

if TRAIN_FLAG:
    trainer.fit(lightning_model, train_loader, val_loader)
else:
    trainer.validate(lightning_model, val_loader)
