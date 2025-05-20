import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import io
from PIL import Image
import wandb
# Use Devanagari for Hindi, and fallback to DejaVu Sans for English
plt.rcParams['font.family'] = ['Noto Sans Devanagari', 'DejaVu Sans']

def pad_tensor_2d(attn, max_tgt, max_src):
    # attn: (tgt_len, src_len)
    tgt_len, src_len = attn.size()
    pad_tgt = max_tgt - tgt_len
    pad_src = max_src - src_len
    pad = (0, pad_src, 0, pad_tgt)
    return torch.nn.functional.pad(attn, pad, "constant", 0)

def get_predictions(model, dataset, dataloader, device, max_length=50):
    model.eval()
    all_attention = []
    all_predictions = []
    all_inputs = []
    
    source_vocab_inv = {v: k for k, v in dataset.source_vocab.items()}
    target_vocab_inv = {v: k for k, v in dataset.target_vocab.items()}
    sos_idx = dataset.target_vocab['<sos>']
    eos_idx = dataset.target_vocab['<eos>']
    pad_idx = dataset.target_vocab['<pad>']

    # First pass: collect all attention shapes
    max_tgt_len = 0
    max_src_len = 0
    attn_maps_per_sample = []
    pred_per_sample = []
    input_per_sample = []

    with torch.no_grad():
        for src_batch, _ in dataloader:
            src_batch = src_batch.to(device)
            if src_batch.dim() == 1:
                src_batch = src_batch.unsqueeze(0)
            batch_size = src_batch.size(0)
            src_lens = (src_batch != pad_idx).sum(dim=1).cpu()
            encoder_outputs, hidden = model.encoder(src_batch, src_lens)
            src_mask = model.create_mask(src_batch)
            
            input_char = torch.full((batch_size,), sos_idx, dtype=torch.long, device=device)
            outputs = []
            attn_weights = []
            model.decoder.reset_attention()
            
            ended = torch.zeros(batch_size, dtype=torch.bool, device=device)
            for step in range(max_length):
                output, hidden = model.decoder(input_char, hidden, encoder_outputs, src_mask)
                top1 = output.argmax(1)
                outputs.append(top1)
                attn_weights.append(model.decoder.attention_weights[-1])
                input_char = top1
                ended = ended | (top1 == eos_idx)
                if ended.all():
                    break
            
            outputs = torch.stack(outputs, dim=1)  # (batch, tgt_len)
            attn_weights = torch.stack(attn_weights, dim=1)  # (batch, tgt_len, src_len)
            
            for i in range(batch_size):
                attn_map = attn_weights[i][:, :encoder_outputs.size(1)]
                tgt_len, src_len = attn_map.size()
                max_tgt_len = max(max_tgt_len, tgt_len)
                max_src_len = max(max_src_len, src_len)
                attn_maps_per_sample.append(attn_map.cpu())
                
                pred_ids = outputs[i].tolist()
                pred_str = []
                for idx in pred_ids:
                    if idx == eos_idx:
                        break
                    if idx in (sos_idx, pad_idx):
                        continue
                    pred_str.append(target_vocab_inv.get(idx, '?'))
                pred_per_sample.append(''.join(pred_str))
                
                input_ids = src_batch[i][src_batch[i] != pad_idx].tolist()
                input_str = ''.join([source_vocab_inv.get(idx, '?') for idx in input_ids])
                input_per_sample.append(input_str)

    # Now pad all attention maps to (max_tgt_len, max_src_len)
    all_attention = [pad_tensor_2d(attn, max_tgt_len, max_src_len) for attn in attn_maps_per_sample]
    all_attention = torch.stack(all_attention, dim=0)
    return all_attention, pred_per_sample, input_per_sample




def plot_attention_grid(attention_weights, input_words, output_words, num_plots=10,example=0):
    """
    Plots attention heatmaps for up to 10 examples in a 3x4 grid.
    Each heatmap uses only the true (unpadded) input and output lengths.
    """
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()
    n = min(num_plots, len(attention_weights), len(input_words), len(output_words))
    for i in range(n):
        attn = attention_weights[i]
        inp = input_words[i]
        out = output_words[i]
        # Only plot up to the true lengths
        attn_plot = attn[:len(out), :len(inp)].numpy()
        sns.heatmap(attn_plot, 
                    annot=False, 
                    xticklabels=list(inp),
                    yticklabels=list(out),
                    ax=axes[i],
                    cbar=False)
        axes[i].set_title(f"Ex {i+1}: {inp} → {out}")
        axes[i].set_xlabel("Input Chars")
        axes[i].set_ylabel("Output Chars")
        # Rotate x labels for clarity
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right')
    # Hide unused subplots
    for j in range(n, 12):
        axes[j].axis('off')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    pil_img = Image.open(buf)
    wandb.log({"example":example,"attention grid": wandb.Image(pil_img)})
    buf.close()
    plt.show()
