import os
import time
import warnings
from typing import Any
from logging import Logger
from tqdm import tqdm
import torch
from nltk.translate.bleu_score import sentence_bleu
from TrajLearn.TrajectoryBatchDataset import TrajectoryBatchDataset

def calculate_bleu(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    bleu_score = 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for prediction, target in zip(predictions, targets):
            prediction = prediction.tolist()
            target = target.tolist()
            bleu_score += sentence_bleu([target], prediction)
    return bleu_score


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataset: TrajectoryBatchDataset,
    config: Any,
    logger: Logger,
    top_k: list = None,
) -> list:
    model.eval()
    device = config["device"]
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    prediction_length = config["test_prediction_length"]
    ctx = torch.amp.autocast(device_type=device_type, dtype=torch.float32)

    if top_k is None:
        top_k = [1, 3, 5]

    beam_width = config["beam_width"]

    if config["continuity"]:
        neighbors = dataset.get_neighbors()

    total_bleu_score = 0.0
    correct_predictions = {k: torch.zeros(
        prediction_length, dtype=torch.int32).to(device) for k in top_k}

    if config["store_predictions"]:
        pred_results_buffer = ["input sequence,true label,predicted label\n"]
        pred_results_file = open(os.path.join(logger.log_directory, 'predictions.txt'), 'w', encoding='utf-8')

    start_time = time.time()

    total_samples = 0
    for X, Y in (pbar := tqdm(dataset, leave=False)):
        x, y = X.to(device), Y.to(device)
        beams = torch.zeros((x.shape[0], 1, 0), dtype=torch.int32).to(device)
        scores = torch.zeros((x.shape[0], 1), dtype=torch.float32).to(device)

        total_samples += x.shape[0]
        for j in range(prediction_length):
            new_scores, new_beams = [], []
            for b in range(beams.shape[1]):
                beam = beams[:, b:b+1]
                with ctx:
                    input_sequence = torch.cat((x, beam.squeeze(1)), dim=1)
                    logits, _ = model(
                        input_sequence[:, -config["block_size"]:])
                    logits = torch.squeeze(logits, dim=1)
                    probs = torch.softmax(logits, dim=1)

                    # Apply the mask
                    if config["continuity"]:
                        last_prediction = input_sequence[:, -1]
                        mask = torch.zeros_like(logits, dtype=torch.bool)
                        for idx, item in enumerate(last_prediction):
                            mask[idx, neighbors[item.item()]] = True
                        probs[~mask] = 0

                    # Get top-k probabilities and their indices
                    top_probs, indices = torch.topk(probs, beam_width)
                    indices = torch.where(
                        indices == 0, torch.ones_like(indices), indices)
                    # Append new indices to beam
                    new_beam = torch.cat(
                        (beam.repeat(1, beam_width, 1), indices.unsqueeze(2)), dim=2)
                    new_score = scores[:, b:b+1] + \
                        torch.log(top_probs)  # Update scores
                    new_scores.append(new_score)
                    new_beams.append(new_beam)
            # Concatenate along beam dimension
            new_scores = torch.cat(new_scores, dim=1)
            # Concatenate along beam dimension
            new_beams = torch.cat(new_beams, dim=1)
            top_scores, top_beams = torch.topk(new_scores.view(
                X.shape[0], -1), beam_width)  # Reshape scores to 2D and get top-k
            # Reshape beams to 3D for gathering
            beams = new_beams.view(X.shape[0], -1, new_beams.shape[2])
            beams = torch.gather(beams, 1, top_beams.unsqueeze(
                2).expand(-1, -1, beams.shape[2]))  # Gather the top-k beams
            scores = top_scores  # Update scores with top-k scores

            for k in correct_predictions.keys():
                if beam_width >= k:
                    predictions = beams[:, :k]  # Get the top-k beams
                    for beam_number in range(k):
                        correct_predictions[k][j] += torch.sum((predictions[:, beam_number:beam_number+1].squeeze(1) ==
                                                                y[:, :j+1]).all(dim=1).int())

        total_bleu_score += calculate_bleu(beams[:, 0], y)

        if config["store_predictions"]:
            beams_np = beams[:, 0].cpu().numpy()
            xs_np, ys_np = x.cpu().numpy(), y.cpu().numpy()
            for sample_id, x_np in enumerate(xs_np):
                prediction_result = f"{' '.join(map(str, x_np))}, {' '.join(map(str, ys_np[sample_id]))}, {' '.join(map(str, beams_np[sample_id]))}\n"
                pred_results_buffer.append(prediction_result)
                if len(pred_results_buffer) > 10000:
                    pred_results_file.writelines(pred_results_buffer)
                    pred_results_buffer = []

        acc1 = ((100 * correct_predictions[1][prediction_length-1]) / total_samples).item()
        pbar.set_postfix(**{"Acc@1": acc1})

    if config["store_predictions"]:
        pred_results_file.writelines(pred_results_buffer)

    test_duration = time.time() - start_time

    avg_bleu_score = total_bleu_score / total_samples
    acc = {k: (100 * v) / (total_samples)
           for k, v in correct_predictions.items()}
    results = [f"Dataset: {config['dataset']}"]
    for k, v in acc.items():
        results.append(f"Accuracy@{k}: {v[-1]:.4f}")
    results.append(f"BLEU score: {avg_bleu_score:.4f}")
    results.append(f"Test duration: {test_duration:.3f}(s)")
    results.append(f"Samples: {total_samples}")
    logger.info(", ".join(results))

    return results
