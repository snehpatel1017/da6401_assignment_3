import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import random
from tqdm import tqdm
import os

# -------------------------------
# Dataset and Vocabulary
# -------------------------------
class TransliterationDataset(Dataset):
    def __init__(self, data, source_vocab=None, target_vocab=None):
        self.pairs = data
        self.source_vocab = source_vocab or self.build_vocab(data['latin'])
        self.target_vocab = target_vocab or self.build_vocab(data['devanagari'], add_sos_eos=True)
        self.inv_target_vocab = {v: k for k, v in self.target_vocab.items()}
        
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        x = list(self.pairs.iloc[idx]['latin'])
        y = list(self.pairs.iloc[idx]['devanagari'])
        x_idx = [self.source_vocab.get(char, self.source_vocab['<unk>']) for char in x]
        y_idx = [self.target_vocab['<sos>']] + [self.target_vocab.get(char, self.target_vocab['<unk>']) for char in y] + [self.target_vocab['<eos>']]
        return torch.tensor(x_idx), torch.tensor(y_idx)
    
    def build_vocab(self, sequences, add_sos_eos=False):
        chars = set(char for seq in sequences for char in seq)
        vocab = {char: idx + 2 for idx, char in enumerate(sorted(chars))}
        vocab['<pad>'] = 0
        vocab['<unk>'] = 1
        if add_sos_eos:
            vocab['<sos>'] = len(vocab)
            vocab['<eos>'] = len(vocab)
        return vocab

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_lens = [len(seq) for seq in src_batch]
    tgt_lens = [len(seq) for seq in tgt_batch]
    src_padded = nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_padded = nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=0)
    return src_padded, tgt_padded, src_lens, tgt_lens

# -------------------------------
# Encoder-Decoder Architecture
# -------------------------------
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, num_layers, cell_type, dropout=0.0):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        rnn_class = getattr(nn, cell_type)
        self.rnn = rnn_class(emb_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)

    def forward(self, src, src_lens):
        embedded = self.embedding(src)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, src_lens, batch_first=True, enforce_sorted=False)
        _, hidden = self.rnn(packed)
        return hidden

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, num_layers, cell_type, dropout=0.0):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        rnn_class = getattr(nn, cell_type)
        self.rnn = rnn_class(emb_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_char, hidden):
        embedded = self.embedding(input_char.unsqueeze(1))
        output, hidden = self.rnn(embedded, hidden)
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        batch_size, tgt_len = tgt.shape
        tgt_vocab_size = self.decoder.embedding.num_embeddings
        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(src.device)
        
        src_lens = [torch.count_nonzero(s).item() for s in src]
        hidden = self.encoder(src, src_lens)
        input_char = tgt[:, 0]

        for t in range(1, tgt_len):
            output, hidden = self.decoder(input_char, hidden)
            outputs[:, t] = output
            top1 = output.argmax(1)
            input_char = tgt[:, t] if random.random() < teacher_forcing_ratio else top1
        return outputs

# -------------------------------
# Training Utilities
# -------------------------------
def calculate_word_accuracy(preds, tgt, pad_idx=0, sos_idx=None, eos_idx=None):
    """
    Computes word-level (exact match) accuracy for a batch.
    Ignores <pad>, <sos>, and <eos> tokens in both prediction and target.
    Args:
        preds: (batch, seq_len, vocab_size) logits or (batch, seq_len) indices
        tgt: (batch, seq_len) indices
        pad_idx: int, index for <pad>
        sos_idx: int or None, index for <sos>
        eos_idx: int or None, index for <eos>
    Returns:
        accuracy: float
    """
    if preds.dim() == 3:
        preds = preds.argmax(dim=2)
    correct = 0
    total = 0
    for pred_seq, tgt_seq in zip(preds, tgt):
        # Remove special tokens from both sequences
        pred_tokens = [p.item() for p in pred_seq
                       if p.item() != pad_idx and
                          (sos_idx is None or p.item() != sos_idx) and
                          (eos_idx is None or p.item() != eos_idx)]
        tgt_tokens = [t.item() for t in tgt_seq
                      if t.item() != pad_idx and
                         (sos_idx is None or t.item() != sos_idx) and
                         (eos_idx is None or t.item() != eos_idx)]
        if pred_tokens == tgt_tokens:
            correct += 1
        total += 1
    return correct / total if total > 0 else 0


def train_model(config=None):
    with wandb.init(config=config,project="da6401_assignment_3", entity="cs24m048-iit-madras"):
        config = wandb.config

        # Load dataset
        lang_code = config.lang_code
        base_path = f'dakshina_dataset_v1.0/{lang_code}/lexicons'
        df_train = pd.read_csv(f"{base_path}/{lang_code}.translit.sampled.train.tsv", sep='\t', names=['devanagari', 'latin', 'people'])[['devanagari', 'latin']].dropna()
        df_val = pd.read_csv(f"{base_path}/{lang_code}.translit.sampled.dev.tsv", sep='\t', names=['devanagari', 'latin', 'people'])[['devanagari', 'latin']].dropna()
        df_test = pd.read_csv(f"{base_path}/{lang_code}.translit.sampled.test.tsv", sep='\t', names=['devanagari', 'latin', 'people'])[['devanagari', 'latin']].dropna()

        base_dataset = TransliterationDataset(df_train)
        dataset_train = TransliterationDataset(df_train, base_dataset.source_vocab, base_dataset.target_vocab)
        dataset_val = TransliterationDataset(df_val, base_dataset.source_vocab, base_dataset.target_vocab)
        dataset_test = TransliterationDataset(df_test, base_dataset.source_vocab, base_dataset.target_vocab)

        train_loader = DataLoader(dataset_train, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(dataset_val, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(dataset_test, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder = Encoder(len(base_dataset.source_vocab), config.embed_dim, config.hidden_dim, config.num_layers, config.cell_type, config.dropout).to(device)
        decoder = Decoder(len(base_dataset.target_vocab), config.embed_dim, config.hidden_dim, config.num_layers, config.cell_type, config.dropout).to(device)
        model = Seq2Seq(encoder, decoder).to(device)

        optimizer = optim.Adam(model.parameters(), lr=config.lr)
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        pad_idx = base_dataset.target_vocab['<pad>']
        sos_idx = base_dataset.target_vocab['<sos>']
        eos_idx = base_dataset.target_vocab['<eos>']

        for epoch in range(1, config.epochs + 1):
            model.train()
            total_loss = 0
            train_acc = 0
            for src, tgt, _, _ in train_loader:
                src, tgt = src.to(device), tgt.to(device)
                optimizer.zero_grad()
                output = model(src, tgt)
               
                train_acc += calculate_word_accuracy(output, tgt, pad_idx=pad_idx, sos_idx=sos_idx, eos_idx=eos_idx)
                
                loss = criterion(output[:, 1:].reshape(-1, output.shape[-1]), tgt[:, 1:].reshape(-1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            train_acc /= len(train_loader)

            model.eval()
            val_acc = 0
            test_acc = 0
            val_loss = 0
            with torch.no_grad():
                for src, tgt, _, _ in val_loader:
                    src, tgt = src.to(device), tgt.to(device)
                    output = model(src, tgt, teacher_forcing_ratio=0.0)
                    val_acc += calculate_word_accuracy(output, tgt, pad_idx=pad_idx, sos_idx=sos_idx, eos_idx=eos_idx)
                    val_loss += criterion(output[:, 1:].reshape(-1, output.shape[-1]), tgt[:, 1:].reshape(-1)).item()
                for src, tgt, _, _ in test_loader:
                    src, tgt = src.to(device), tgt.to(device)
                    output = model(src, tgt, teacher_forcing_ratio=0.0)
                    test_acc += calculate_word_accuracy(output, tgt, pad_idx=pad_idx, sos_idx=sos_idx, eos_idx=eos_idx)
            val_acc /= len(val_loader)
            test_acc /= len(test_loader)
            wandb.log({"epoch": epoch, "loss": total_loss / len(train_loader), "val_accuracy": val_acc , "test_accuracy":test_acc,"train_accuracy":train_acc , "val_loss":val_loss})
            print(f"Epoch {epoch}, Loss: {total_loss / len(train_loader)}, Train Accuracy: {train_acc}, Val Accuracy: {val_acc}, Test Accuracy: {test_acc}, Validation loss: {val_loss}")
        return (model , dataset_test , test_loader , device)
            

