import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import random
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Attention Mechanism
# -------------------------------
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, src_mask):
        src_len = encoder_outputs.size(1)
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        attention = attention.masked_fill(src_mask == 0, -1e10)
        return torch.softmax(attention, dim=1)

# -------------------------------
# Dataset and Vocabulary
# -------------------------------
class TransliterationDataset(Dataset):
    def __init__(self, data, source_vocab=None, target_vocab=None):
        # Clean data: remove NaN entries and convert to strings
        self.data = data.dropna(subset=['latin', 'devanagari'])
        self.data['latin'] = self.data['latin'].astype(str)
        self.data['devanagari'] = self.data['devanagari'].astype(str)
        
        # Build vocabularies
        self.source_vocab = source_vocab or self.build_vocab(self.data['latin'])
        self.target_vocab = target_vocab or self.build_vocab(self.data['devanagari'], add_sos_eos=True)
        self.inv_target_vocab = {v: k for k, v in self.target_vocab.items()}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        x = str(row['latin']).strip()
        y = str(row['devanagari']).strip()
        x_idx = [self.source_vocab.get(char, self.source_vocab['<unk>']) for char in x]
        y_idx = [self.target_vocab['<sos>']] + \
                [self.target_vocab.get(char, self.target_vocab['<unk>']) for char in y] + \
                [self.target_vocab['<eos>']]
        return torch.tensor(x_idx), torch.tensor(y_idx)

    def build_vocab(self, sequences, add_sos_eos=False):
        valid_seqs = [str(seq) for seq in sequences if pd.notna(seq) and str(seq).strip()]
        chars = set()
        for seq in valid_seqs:
            for char in seq.strip():
                chars.add(char)
        vocab = {'<pad>': 0, '<unk>': 1}
        for idx, char in enumerate(sorted(chars)):
            vocab[char] = idx + 2  # Start from index 2
        if add_sos_eos:
            vocab['<sos>'] = len(vocab)
            vocab['<eos>'] = len(vocab)
        return vocab

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_padded = nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_padded = nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=0)
    return src_padded, tgt_padded

# -------------------------------
# Encoder
# -------------------------------
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, num_layers, cell_type, dropout=0.0):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        rnn_class = getattr(nn, cell_type)
        self.rnn = rnn_class(emb_dim, hidden_dim, num_layers, 
                           batch_first=True, 
                           dropout=dropout if num_layers > 1 else 0.0)

    def forward(self, src, src_lens):
        embedded = self.embedding(src)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, src_lens, 
                                                 batch_first=True, 
                                                 enforce_sorted=False)
        packed_output, hidden = self.rnn(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        return output, hidden

# -------------------------------
# Decoder with Attention Tracking
# -------------------------------
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, num_layers, cell_type, dropout=0.0, attention=None):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        rnn_class = getattr(nn, cell_type)
        self.rnn = rnn_class(emb_dim + hidden_dim, hidden_dim, num_layers,
                           batch_first=True, 
                           dropout=dropout if num_layers > 1 else 0.0)
        self.fc_out = nn.Linear(hidden_dim * 2, output_dim)
        self.attention = attention
        self.attention_weights = []

    def forward(self, input_char, hidden, encoder_outputs, src_mask):
        embedded = self.embedding(input_char.unsqueeze(1))
        h = hidden[0][-1] if isinstance(hidden, tuple) else hidden[-1]
        attn_weights = self.attention(h, encoder_outputs, src_mask)
        if not self.training:
            self.attention_weights.append(attn_weights.detach().cpu())
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        rnn_input = torch.cat((embedded, context), dim=2)
        output, hidden = self.rnn(rnn_input, hidden)
        prediction = self.fc_out(torch.cat(
            (output.squeeze(1), context.squeeze(1)), dim=1
        ))
        return prediction, hidden

    def reset_attention(self):
        self.attention_weights = []

# -------------------------------
# Seq2Seq Model with Visualization Support
# -------------------------------
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def create_mask(self, seq):
        return (seq != 0).float()

    def forward(self, src, tgt=None, teacher_forcing_ratio=0.5, return_attention=False, max_length=50):
        # Handle inference case (no tgt provided)
        if tgt is None:
            return self.infer(src, max_length, return_attention)

        # Original training logic
        batch_size, tgt_len = tgt.shape
        tgt_vocab_size = self.decoder.embedding.num_embeddings
        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(src.device)
        
        if return_attention:
            self.decoder.reset_attention()
        
        src_lens = torch.count_nonzero(src, dim=1).cpu()
        encoder_outputs, hidden = self.encoder(src, src_lens)
        src_mask = self.create_mask(src)
        
        input_char = tgt[:, 0]
        for t in range(1, tgt_len):
            output, hidden = self.decoder(input_char, hidden, encoder_outputs, src_mask)
            outputs[:, t] = output
            top1 = output.argmax(1)
            input_char = tgt[:, t] if random.random() < teacher_forcing_ratio else top1
        
        if return_attention:
            attn_weights = torch.stack(self.decoder.attention_weights, dim=1)
            return outputs, attn_weights
        return outputs


    def infer(self, src, max_length=50, return_attention=False):
        """Fixed inference method with dimension handling"""
        # Ensure input is 2D (batch, seq_len)
        if src.dim() == 1:
            src = src.unsqueeze(0)  # Add batch dimension
        
        batch_size = src.size(0)
        tgt_vocab_size = self.decoder.embedding.num_embeddings
        outputs = torch.zeros(batch_size, max_length, tgt_vocab_size).to(src.device)
        
        if return_attention:
            self.decoder.reset_attention()
        
        src_lens = (src != 0).sum(dim=1).cpu()  # Now safe with 2D input
        encoder_outputs, hidden = self.encoder(src, src_lens)
        src_mask = self.create_mask(src)
        
        input_char = torch.full((batch_size,), 
                              self.decoder.embedding.padding_idx + 1,  # <sos>
                              device=src.device)
        
        for t in range(max_length):
            output, hidden = self.decoder(input_char, hidden, encoder_outputs, src_mask)
            outputs[:, t] = output
            input_char = output.argmax(1)
        
        if return_attention:
            attn_weights = torch.stack(self.decoder.attention_weights, dim=1)
            return outputs, attn_weights
        return outputs


# -------------------------------
# Accuracy Calculation
# -------------------------------

def calculate_word_accuracy(model, dataloader, device):
    """
    Computes word-level (exact match) accuracy for your seq2seq model.
    Only counts as correct if the entire predicted word matches the target word,
    ignoring <pad>, <sos>, and <eos> tokens.
    """
    model.eval()
    total_correct = 0
    total_words = 0

    # Get special token indices from the dataset's target vocab
    vocab = dataloader.dataset.target_vocab
    pad_idx = vocab['<pad>']
    sos_idx = vocab['<sos>']
    eos_idx = vocab['<eos>']

    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            outputs = model(src, tgt)  # (batch, seq_len, vocab_size)
            preds = outputs.argmax(-1) # (batch, seq_len)
            for pred_seq, tgt_seq in zip(preds, tgt):
                # Remove special tokens from both sequences
                pred_tokens = [p.item() for p in pred_seq
                               if p.item() not in {pad_idx, sos_idx, eos_idx}]
                tgt_tokens = [t.item() for t in tgt_seq
                              if t.item() not in {pad_idx, sos_idx, eos_idx}]
                if pred_tokens == tgt_tokens:
                    total_correct += 1
                total_words += 1
    return total_correct / total_words if total_words > 0 else 0.0


# -------------------------------
# Training Loop with Validation
# -------------------------------
def train_model(config):
    # Load training and validation data
    base_path = f'dakshina_dataset_v1.0/{config.lang_code}/lexicons'
    df_train = pd.read_csv(
        f"{base_path}/{config.lang_code}.translit.sampled.train.tsv",
        sep='\t',
        names=['devanagari', 'latin', 'people'],
        usecols=['devanagari', 'latin'],
        dtype={'devanagari': str, 'latin': str}
    ).dropna().reset_index(drop=True)
    df_val = pd.read_csv(
        f"{base_path}/{config.lang_code}.translit.sampled.dev.tsv",
        sep='\t',
        names=['devanagari', 'latin', 'people'],
        usecols=['devanagari', 'latin'],
        dtype={'devanagari': str, 'latin': str}
    ).dropna().reset_index(drop=True)
    df_test = pd.read_csv(
        f"{base_path}/{config.lang_code}.translit.sampled.test.tsv",
        sep='\t',
        names=['devanagari', 'latin', 'people'],
        usecols=['devanagari', 'latin'],
        dtype={'devanagari': str, 'latin': str}
    ).dropna().reset_index(drop=True)

    train_dataset = TransliterationDataset(df_train)
    val_dataset = TransliterationDataset(df_val, train_dataset.source_vocab, train_dataset.target_vocab)
    test_dataset = TransliterationDataset(df_test, train_dataset.source_vocab, train_dataset.target_vocab)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                            shuffle=True, collate_fn=collate_fn, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                          shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                          shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=False)

    # Model initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attention = Attention(config.hidden_dim)
    encoder = Encoder(len(train_dataset.source_vocab), config.embed_dim, 
                    config.hidden_dim, config.num_layers, config.cell_type)
    decoder = Decoder(len(train_dataset.target_vocab), config.embed_dim,
                    config.hidden_dim, config.num_layers, config.cell_type, 
                    attention=attention)
    model = Seq2Seq(encoder, decoder).to(device)

    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    best_val_acc = 0.0

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        for src, tgt in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}"):
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            outputs = model(src, tgt)
            loss = criterion(outputs[:, 1:].reshape(-1, outputs.shape[-1]),
                             tgt[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
       
        train_acc = calculate_accuracy(model, train_loader, device)
        val_acc = calculate_accuracy(model, val_loader, device)
        test_acc = calculate_accuracy(model, test_loader, device)
        wandb.log({"epoch": epoch, "loss": total_loss / len(train_loader), "val_accuracy": val_acc , "test_accuracy":test_acc,"train_accuracy":train_acc , "val_loss":total_loss / len(train_loader)})
        print(f"Epoch {epoch+1}/{config.epochs} | Loss: {total_loss/len(train_loader):.4f} | "
              f"Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}% Test Acc: {test_acc*100:.2f}%")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
    print(f'Best Validation Accuracy: {best_val_acc*100:.2f}%')
    return model, train_dataset, val_dataset, device
