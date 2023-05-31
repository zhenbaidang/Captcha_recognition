import torch
import Levenshtein as lev
from data import get_dict

def transposition(output):
    # output.shape = (weight(seq_len), batch_size, num_class)
    _, max_indices = torch.max(output, dim=2) # shape: (weight(seq_len), batch_size)

    # Initialize list to hold the decoded outputs
    decoded_outputs = []

    # For each sequence in the batch
    for i in range(max_indices.shape[1]):

        # Get the predicted sequence for this batch element
        seq = max_indices[:, i]

        # Remove duplicates and blank labels
        decoded_seq = []
        prev_elem = None
        for idx in seq:
            idx = idx.item()
            if idx != 0 and idx != prev_elem:  # assuming blank label is 0
                decoded_seq.append(idx)
            prev_elem = idx

        decoded_outputs.append(decoded_seq)

    return decoded_outputs # [[]] batch_size, seq_len(注意转录生成的预测长度是参差不齐的)

def convert2string_list(batch_index):
    if isinstance(batch_index, torch.Tensor):
        batch_index = batch_index.tolist()
    elif isinstance(batch_index, list):
        pass
    else:
        raise TypeError
    _, id2label = get_dict()
    return [''.join([id2label[char_] for char_ in line]) for line in batch_index]

def acc(predict_decoded, labels):
    # predict_decoded: list of list
    # Assuming you have a list of ground truth labels and a list of predicted labels
    predict_string = convert2string_list(predict_decoded)
    labels_string = convert2string_list(labels)
    

    distances = [lev.distance(t, p) for t, p in zip(labels_string, predict_string)]
    total_characters = sum(len(t) for t in labels_string)
    # Character Error Rate (CER)
    # CER = (Substitutions + Deletions + Insertions) / Total number of reference characters
    cer = sum(distances) / total_characters

    all_eq = [t == p for t, p in zip(labels_string, predict_string)]
    multi_acc = sum(all_eq) / len(all_eq)

    return 1 - cer, multi_acc
