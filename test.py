import os
import json
from nltk.translate.bleu_score import sentence_bleu
import warnings
from tqdm import tqdm
import torch


def calculate_bleu(predictions, targets):
    bleu_score = 0.0
    for prediction, target in zip(predictions, targets):
        # Converting tensors to lists
        prediction = prediction.tolist()
        target = target.tolist()
        # Compute BLEU score
        # TODO: Fix this warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            score = sentence_bleu([target], prediction)
            bleu_score += score

    return bleu_score


@torch.no_grad()
def test(model, dataset, config, logger, beam_size=1):
    model.eval()
    device = config["device"]
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    prediction_length = config["test_prediction_length"]
    dtype = 'float32'
    ptdtype = {
        'float32': torch.float32,
        'bfloat16': torch.bfloat16,
        'float16': torch.float16}[dtype]
    ctx = torch.amp.autocast(
        device_type=device_type, dtype=ptdtype)

    neighbors = dataset.get_neighbors()

    total_bleu_score = 0.0
    true_predictions = {1: torch.zeros((prediction_length), dtype=torch.int32).to(device),
                        3: torch.zeros((prediction_length), dtype=torch.int32).to(device),
                        5: torch.zeros((prediction_length), dtype=torch.int32).to(device)}

    dataset.create_batches(
        config["batch_size"], config["test_input_length"], prediction_length, False, False)

    for X, Y in tqdm(dataset, leave=False):
        x = X.to(device)
        y = Y.to(device)
        beams = torch.zeros((x.shape[0], 1, 0), dtype=torch.int32).to(device)
        scores = torch.zeros((x.shape[0], 1), dtype=torch.float32).to(device)

        for j in range(prediction_length):
            new_scores = []
            new_beams = []
            for b in range(beams.shape[1]):
                beam = beams[:, b:b+1]
                with ctx:
                    logits, _ = model(
                        torch.cat((x, beam.squeeze(1)), dim=1))
                    logits = torch.squeeze(logits, dim=1)

                    # Get the last prediction in the beam
                    last_prediction = beam[:, :, -
                                           1] if beam.size(2) > 0 else x[:, -1]
                    # Apply the mask
                    mask = torch.zeros_like(logits, dtype=torch.bool)
                    for idx, item in enumerate(last_prediction):
                        mask[idx, neighbors[item.item()]] = True

                    probs = torch.softmax(logits, dim=1)
                    probs[~mask] = 0
                    # Get top-k probabilities and their indices
                    top_probs, indices = torch.topk(probs, beam_size)
                    indices = torch.where(
                        indices == 0, torch.ones_like(indices), indices)
                    # Append new indices to beam
                    new_beam = torch.cat(
                        (beam.repeat(1, beam_size, 1), indices.unsqueeze(2)), dim=2)
                    new_score = scores[:, b:b+1] + \
                        torch.log(top_probs)  # Update scores
                    new_scores.append(new_score)
                    new_beams.append(new_beam)
            # Concatenate along beam dimension
            new_scores = torch.cat(new_scores, dim=1)
            # Concatenate along beam dimension
            new_beams = torch.cat(new_beams, dim=1)
            top_scores, top_beams = torch.topk(new_scores.view(
                X.shape[0], -1), beam_size)  # Reshape scores to 2D and get top-k
            # Reshape beams to 3D for gathering
            beams = new_beams.view(X.shape[0], -1, new_beams.shape[2])
            beams = torch.gather(beams, 1, top_beams.unsqueeze(
                2).expand(-1, -1, beams.shape[2]))  # Gather the top-k beams
            scores = top_scores  # Update scores with top-k scores

            for k in true_predictions.keys():
                if beam_size >= k:
                    predictions = beams[:, :k]  # Get the top-k beams
                    for p in range(k):
                        true_predictions[k][j] += torch.sum((predictions[:, p:p+1].squeeze(1) ==
                                                            y[:, :j+1]).all(dim=1).int())
        total_bleu_score += calculate_bleu(beams[:, 0], y)
    avg_bleu_score = total_bleu_score / len(dataset)
    acc = {k: v / len(dataset) for k, v in true_predictions.items()}
    results = []
    for k, v in acc.items():
        results.append(f"Accuracy@{k}: {v[-1]:.4f}")
    logger.info(", ".join(results) + ", " +
                f"BLEU score: {avg_bleu_score:.4f}")
    return acc, avg_bleu_score
