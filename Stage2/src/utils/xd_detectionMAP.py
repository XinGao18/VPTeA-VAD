import numpy as np

from utils.tools import XD_ANOMALY_CLASSES, XD_CLASS_NAMES


def smooth(values):
    return values


def nms(dets, thresh=0.6, top_k=-1):
    """Pure Python NMS baseline."""
    if len(dets) == 0:
        return []
    order = np.arange(0, len(dets), 1)
    dets = np.array(dets)
    x1 = dets[:, 0]
    x2 = dets[:, 1]
    lengths = x2 - x1
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if len(keep) == top_k:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1)
        overlap = inter / (lengths[i] + lengths[order[1:]] - inter)
        inds = np.where(overlap <= thresh)[0]
        order = order[inds + 1]

    return dets[keep], keep


def getLocMAP(predictions, th, gtsegments, gtlabels, excludeNormal):
    if excludeNormal:
        classlist = list(XD_ANOMALY_CLASSES)
        anomaly_indices = [
            index for index, labels in enumerate(gtlabels)
            if any(label in XD_ANOMALY_CLASSES for label in labels)
        ]
        predictions = [predictions[index] for index in anomaly_indices]
        gtsegments = [gtsegments[index] for index in anomaly_indices]
        gtlabels = [gtlabels[index] for index in anomaly_indices]
    else:
        classlist = list(XD_CLASS_NAMES)

    if any(prediction.ndim != 2 or prediction.shape[1] != len(classlist) for prediction in predictions):
        raise ValueError(f'Each prediction must have shape [frames, {len(classlist)}]')
    if len(predictions) != len(gtsegments) or len(predictions) != len(gtlabels):
        raise ValueError(
            f'Prediction and ground-truth counts differ: predictions={len(predictions)}, '
            f'segments={len(gtsegments)}, labels={len(gtlabels)}'
        )

    predictions_mod = []
    class_scores = []
    for prediction in predictions:
        sorted_scores = -prediction.copy()
        [sorted_scores[:, index].sort() for index in range(sorted_scores.shape[1])]
        sorted_scores = -sorted_scores
        topk = max(1, int(sorted_scores.shape[0] / 16))
        class_score = np.mean(sorted_scores[:topk], axis=0)
        class_scores.append(class_score)
        predictions_mod.append(prediction * (class_score > 0.0))
    predictions = predictions_mod

    average_precisions = []
    for class_index, class_name in enumerate(classlist):
        segment_predictions = []
        for video_index, prediction in enumerate(predictions):
            scores = smooth(prediction[:, class_index])
            proposals = []
            for threshold_ratio in np.arange(0.6, 0.7, 0.1):
                threshold = np.max(scores) - (np.max(scores) - np.min(scores)) * threshold_ratio
                binary = np.concatenate([
                    np.zeros(1),
                    (scores > threshold).astype('float32'),
                    np.zeros(1)
                ])
                differences = [binary[index] - binary[index - 1] for index in range(1, len(binary))]
                starts = [index for index, value in enumerate(differences) if value == 1]
                ends = [index for index, value in enumerate(differences) if value == -1]
                for start, end in zip(starts, ends):
                    if end - start >= 2:
                        score = np.max(scores[start:end]) + 0.7 * class_scores[video_index][class_index]
                        proposals.append([video_index, start, end, score])
            if proposals:
                proposals = np.array(proposals)
                proposals = proposals[np.argsort(-proposals[:, -1])]
                _, keep = nms(proposals[:, 1:-1], 0.6)
                segment_predictions.extend(list(proposals[keep]))

        if not segment_predictions:
            average_precisions.append(0.0)
            continue
        segment_predictions = np.array(segment_predictions)
        segment_predictions = segment_predictions[np.argsort(-segment_predictions[:, 3])]

        segment_ground_truth = [
            [video_index, gtsegments[video_index][segment_index][0], gtsegments[video_index][segment_index][1]]
            for video_index in range(len(gtsegments))
            for segment_index in range(len(gtsegments[video_index]))
            if gtlabels[video_index][segment_index] == class_name
        ]
        ground_truth_positives = len(segment_ground_truth)
        if ground_truth_positives == 0:
            average_precisions.append(0.0)
            continue

        true_positives = []
        false_positives = []
        for prediction in segment_predictions:
            best_iou = 0.0
            best_index = None
            for ground_truth_index, ground_truth in enumerate(segment_ground_truth):
                if prediction[0] != ground_truth[0]:
                    continue
                ground_truth_range = range(int(ground_truth[1]), int(ground_truth[2]))
                prediction_range = range(int(prediction[1]), int(prediction[2]))
                intersection = set(ground_truth_range).intersection(prediction_range)
                union = set(ground_truth_range).union(prediction_range)
                iou = float(len(intersection)) / float(len(union))
                if iou >= th and iou > best_iou:
                    best_iou = iou
                    best_index = ground_truth_index
            matched = float(best_index is not None)
            if best_index is not None:
                del segment_ground_truth[best_index]
            true_positives.append(matched)
            false_positives.append(1.0 - matched)

        true_positives_cumulative = np.cumsum(true_positives)
        false_positives_cumulative = np.cumsum(false_positives)
        if sum(true_positives) == 0:
            precision = 0.0
        else:
            precision = np.sum(
                true_positives_cumulative
                / (false_positives_cumulative + true_positives_cumulative)
                * true_positives
            ) / ground_truth_positives
        average_precisions.append(precision)

    return 100 * np.mean(average_precisions)


def getDetectionMAP(predictions, segments, labels, excludeNormal=False):
    iou_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    dmap_list = [
        getLocMAP(predictions, iou, segments, labels, excludeNormal)
        for iou in iou_list
    ]
    return dmap_list, iou_list
