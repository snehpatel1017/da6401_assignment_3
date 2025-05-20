import matplotlib.pyplot as plt
import numpy as np
import io
from PIL import Image
import wandb

def plot_connectivity(attention_weights, input_words, output_words, index=0, figsize=(12, 4),example=0):
    """
    Visualize which input character the model attends to most for each output character.
    Handles batched attention weights by selecting the example at `index`.
    - attention_weights: (batch_size, output_len, input_len) numpy array or tensor
    - input_words: list of input strings (batch)
    - output_words: list of output strings (batch)
    - index: which example in the batch to plot
    """
    # Extract the attention matrix for the selected example
    if hasattr(attention_weights, 'cpu'):
        attn = attention_weights[index].cpu().numpy()
    elif isinstance(attention_weights, np.ndarray):
        attn = attention_weights[index]
    else:
        attn = np.array(attention_weights[index])
    
    input_word = input_words[index]
    output_word = output_words[index]

    # Use only the relevant part of the attention matrix
    attn_cropped = attn[:len(output_word), :len(input_word)]

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(attn_cropped, cmap='Blues', alpha=0.3, aspect='auto')

    # Set ticks and labels to match the actual word lengths
    ax.set_xticks(range(len(input_word)))
    ax.set_yticks(range(len(output_word)))
    ax.set_xticklabels(list(input_word), fontsize=14)
    ax.set_yticklabels(list(output_word), fontsize=14)
    ax.set_xlabel("Input sequence", fontsize=16)
    ax.set_ylabel("Output sequence", fontsize=16)
    ax.set_title(f"Connectivity visualization for example {index}", fontsize=16)

    # For each output character, find the input character with max attention
    max_indices = attn_cropped.argmax(axis=1)  # shape: (output_len,)

    for out_i, in_i in enumerate(max_indices):
        ax.plot([in_i], [out_i], 'ro')
        ax.plot([in_i, in_i], [out_i-0.4, out_i+0.4], 'r-', alpha=0.5)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    pil_img = Image.open(buf)
    wandb.log({"example":example ,"connectity grid": wandb.Image(pil_img)})
    buf.close()
    plt.show()
