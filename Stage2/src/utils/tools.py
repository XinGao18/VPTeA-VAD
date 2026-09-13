import json
from collections import Counter

import torch
from torch.utils.data import WeightedRandomSampler
import torch.nn.functional as F
import numpy as np

UCF_CLASS_NAMES = (
    'Normal', 'Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary',
    'Explosion', 'Fighting', 'RoadAccidents', 'Robbery', 'Shooting',
    'Shoplifting', 'Stealing', 'Vandalism'
)
UCF_ANOMALY_CLASSES = UCF_CLASS_NAMES[1:]


def load_ucf_prompt_pairs(path):
    with open(path, encoding='utf-8') as file:
        knowledge_base = json.load(file)

    expected = set(UCF_ANOMALY_CLASSES)
    if set(knowledge_base) != expected:
        missing = sorted(expected - set(knowledge_base))
        unknown = sorted(set(knowledge_base) - expected)
        raise ValueError(f'UCF knowledge base classes mismatch; missing={missing}, unknown={unknown}')

    positive = []
    negative = []
    for class_name in UCF_ANOMALY_CLASSES:
        pair = knowledge_base[class_name]
        if not isinstance(pair, dict) or not isinstance(pair.get('positive'), str) or not isinstance(pair.get('negative'), str):
            raise ValueError(f'Knowledge base entry for {class_name} must contain string positive and negative prompts')
        positive.append(pair['positive'])
        negative.append(pair['negative'])
    return positive, negative


XD_CLASS_NAMES = ('A', 'B1', 'B2', 'B4', 'B5', 'B6', 'G')
XD_ANOMALY_CLASSES = XD_CLASS_NAMES[1:]


def load_xd_prompt_pairs(path):
    with open(path, encoding='utf-8') as file:
        knowledge_base = json.load(file)

    expected = set(XD_ANOMALY_CLASSES)
    if set(knowledge_base) != expected:
        missing = sorted(expected - set(knowledge_base))
        unknown = sorted(set(knowledge_base) - expected)
        raise ValueError(f'XD knowledge base classes mismatch; missing={missing}, unknown={unknown}')

    positive = []
    negative = []
    for class_name in XD_ANOMALY_CLASSES:
        pair = knowledge_base[class_name]
        if not isinstance(pair, dict) or not isinstance(pair.get('positive'), str) or not isinstance(pair.get('negative'), str):
            raise ValueError(f'Knowledge base entry for {class_name} must contain string positive and negative prompts')
        positive.append(pair['positive'])
        negative.append(pair['negative'])
    return positive, negative


def build_class_balanced_sampler(labels, power=0.5):
    if power < 0:
        raise ValueError('Sampling power must be non-negative')
    if not labels:
        raise ValueError('Cannot build a sampler from empty labels')

    counts = Counter(labels)
    weights = torch.tensor(
        [counts[label] ** (-power) for label in labels],
        dtype=torch.double
    )
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(labels),
        replacement=True
    )
    return sampler, weights, counts


def get_batch_pair_label(texts):
    labels = torch.zeros(len(texts), len(UCF_ANOMALY_CLASSES))
    class_indices = {class_name: index for index, class_name in enumerate(UCF_ANOMALY_CLASSES)}
    for row, text in enumerate(texts):
        if text == UCF_CLASS_NAMES[0]:
            continue
        if text not in class_indices:
            raise ValueError(f'Unknown UCF class label: {text}')
        labels[row, class_indices[text]] = 1
    return labels


def get_xd_batch_pair_label(texts):
    labels = torch.zeros(len(texts), len(XD_ANOMALY_CLASSES))
    class_indices = {class_name: index for index, class_name in enumerate(XD_ANOMALY_CLASSES)}
    for row, text in enumerate(texts):
        for class_name in str(text).split('-'):
            if class_name in ('', '0', XD_CLASS_NAMES[0]):
                continue
            if class_name not in class_indices:
                raise ValueError(f'Unknown XD class label: {class_name}')
            labels[row, class_indices[class_name]] = 1
    return labels


def topk_mil_loss(logits, labels, lengths):
    instance_logits = _pool_topk_logits(logits, lengths)
    return F.binary_cross_entropy_with_logits(instance_logits, labels.to(logits.device))


def _pool_topk_logits(logits, lengths):
    instance_logits = []
    for i in range(logits.shape[0]):
        valid_length = min(max(1, int(lengths[i])), logits.shape[1])
        topk = min(max(1, valid_length // 16 + 1), valid_length)
        selected, _ = torch.topk(logits[i, :valid_length], k=topk, largest=True, dim=0)
        instance_logits.append(selected.mean(dim=0))
    return torch.stack(instance_logits, dim=0)


def pair_mil_loss(logits, labels, lengths, positive_weight=4.0, ce_weight=1.0):
    instance_logits = _pool_topk_logits(logits, lengths)
    labels = labels.to(logits.device)
    weights = torch.where(labels > 0, positive_weight, 1.0).to(logits.device)
    bce_loss = F.binary_cross_entropy_with_logits(instance_logits, labels, weight=weights)

    anomaly_mask = labels.sum(dim=1) > 0
    if anomaly_mask.any():
        anomaly_labels = labels[anomaly_mask]
        targets = anomaly_labels / anomaly_labels.sum(dim=1, keepdim=True)
        ce_loss = -(targets * F.log_softmax(instance_logits[anomaly_mask], dim=1)).sum(dim=1).mean()
    else:
        ce_loss = instance_logits.new_zeros(())
    return bce_loss + ce_weight * ce_loss


def _pool_pair_scores(logits, lengths):
    video_scores = []
    for i in range(logits.shape[0]):
        valid_length = min(max(1, int(lengths[i])), logits.shape[1])
        frame_scores = logits[i, :valid_length].max(dim=-1).values
        topk = min(max(1, valid_length // 16 + 1), valid_length)
        selected, _ = torch.topk(frame_scores, k=topk, largest=True)
        video_scores.append(selected.mean())
    return torch.stack(video_scores)


def pair_ranking_loss(logits, labels, lengths, margin=0.2):
    video_scores = _pool_pair_scores(logits, lengths)
    labels = labels.to(logits.device)
    normal_scores = video_scores[labels.sum(dim=1) == 0]
    anomaly_scores = video_scores[labels.sum(dim=1) > 0]
    if normal_scores.numel() == 0 or anomaly_scores.numel() == 0:
        return video_scores.new_zeros(())

    pairwise_margin = margin - anomaly_scores.unsqueeze(1) + normal_scores.unsqueeze(0)
    return F.softplus(pairwise_margin).mean()


def get_batch_label(texts, prompt_text, label_map: dict):
    label_vectors = torch.zeros(0)
    if len(label_map) != 7:
        if len(label_map) == 2:
            for text in texts:
                label_vector = torch.zeros(2)
                if text == 'Normal':
                    label_vector[0] = 1
                else:
                    label_vector[1] = 1
                label_vector = label_vector.unsqueeze(0)
                label_vectors = torch.cat([label_vectors, label_vector], dim=0)
        else:
            for text in texts:
                label_vector = torch.zeros(len(prompt_text))
                if text in label_map:
                    label_text = label_map[text]
                    label_vector[prompt_text.index(label_text)] = 1

                label_vector = label_vector.unsqueeze(0)
                label_vectors = torch.cat([label_vectors, label_vector], dim=0)
    else:
        for text in texts:
            label_vector = torch.zeros(len(prompt_text))
            labels = text.split('-')
            for label in labels:
                if label in label_map:
                    label_text = label_map[label]
                    label_vector[prompt_text.index(label_text)] = 1
            
            label_vector = label_vector.unsqueeze(0)
            label_vectors = torch.cat([label_vectors, label_vector], dim=0)

    return label_vectors

def get_prompt_text(label_map: dict):
    prompt_text = []
    for v in label_map.values():
        prompt_text.append(v)

    return prompt_text

def get_batch_mask(lengths, maxlen):
    batch_size = lengths.shape[0]
    mask = torch.empty(batch_size, maxlen)
    mask.fill_(0)
    for i in range(batch_size):
        if lengths[i] < maxlen:
            mask[i, lengths[i]:maxlen] = 1
    
    return mask.bool()

def random_extract(feat, t_max):
   r = np.random.randint(feat.shape[0] - t_max)
   return feat[r : r+t_max, :]

def uniform_extract(feat, t_max, avg: bool = True):
    new_feat = np.zeros((t_max, feat.shape[1])).astype(np.float32)
    r = np.linspace(0, len(feat), t_max+1, dtype=np.int32)
    if avg == True:
        for i in range(t_max):
            if r[i]!=r[i+1]:
                new_feat[i,:] = np.mean(feat[r[i]:r[i+1],:], 0)
            else:
                new_feat[i,:] = feat[r[i],:]
    else:
        r = np.linspace(0, feat.shape[0]-1, t_max, dtype=np.uint16)
        new_feat = feat[r, :]
            
    return new_feat

def pad(feat, min_len):
    clip_length = feat.shape[0]
    if clip_length <= min_len:
       return np.pad(feat, ((0, min_len - clip_length), (0, 0)), mode='constant', constant_values=0)
    else:
       return feat

def process_feat(feat, length, is_random=False):
    clip_length = feat.shape[0]
    if feat.shape[0] > length:
        if is_random:
            return random_extract(feat, length), length
        else:
            return uniform_extract(feat, length), length
    else:
        return pad(feat, length), clip_length

def process_split(feat, length):
    clip_length = feat.shape[0]
    if clip_length < length:
        return pad(feat, length), clip_length
    else:
        split_num = int(clip_length / length) + 1
        for i in range(split_num):
            if i == 0:
                split_feat = feat[i*length:i*length+length, :].reshape(1, length, feat.shape[1])
            elif i < split_num - 1:
                split_feat = np.concatenate([split_feat, feat[i*length:i*length+length, :].reshape(1, length, feat.shape[1])], axis=0)
            else:
                split_feat = np.concatenate([split_feat, pad(feat[i*length:i*length+length, :], length).reshape(1, length, feat.shape[1])], axis=0)

        return split_feat, clip_length