import torch

def transposition(output):
    # output.shape = (weight(seq_len), batch_size, num_class)
    _, max_indices = torch.max(output, dim=2) # shape: (weight(seq_len), batch_size)
    decoded_outputs = []
    # Initialize list to hold the decoded outputs
    decoded_outputs = []

    # For each sequence in the batch
    for i in range(max_indices.shape[1]):

        # Get the predicted sequence for this batch element
        seq = max_indices[:, i]

        # Remove duplicates and blank labels
        decoded_seq = [seq[0].item()] # 先把序列中的第一个字符index放入数组
        for idx in seq[1:]: # 从第二个字符开始依次遍历
            if idx != decoded_seq[-1] and idx != 0:  # assuming blank label is 0
                decoded_seq.append(idx.item())

        # Convert indices to characters and join them into a string
        str_output = ''.join([alphabet[i-1] for i in decoded_seq])  # assuming alphabet indices start from 1
        decoded_outputs.append(str_output)


def acc()