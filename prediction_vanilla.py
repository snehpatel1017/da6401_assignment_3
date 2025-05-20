def predict_single(model, src_idx, dataset, device, max_len=50):
    """
    Greedy decode a single input sequence using your RNN/GRU/LSTM-based decoder.
    """
    model.eval()
    with torch.no_grad():
        src_tensor = torch.tensor(src_idx, dtype=torch.long).unsqueeze(0).to(device)  # (1, src_len)
        src_len = [len(src_idx)]
        hidden = model.encoder(src_tensor, src_len)  # returns hidden state

        input_char = torch.tensor([dataset.target_vocab['<sos>']], device=device)  # (1,)
        output_idxs = []

        for _ in range(max_len):
            logits, hidden = model.decoder(input_char, hidden)
            top1 = logits.argmax(1).item()
            if top1 == dataset.target_vocab['<eos>']:
                break
            output_idxs.append(top1)
            input_char = torch.tensor([top1], device=device)

    return output_idxs


def log_test_predictions(model, dataset, test_loader, device, n_samples=20):
    import wandb
    data_rows = []
    source_vocab_inv = {v: k for k, v in dataset.source_vocab.items()}
    inv_target_vocab = {v: k for k, v in dataset.target_vocab.items()}
    collected = 0

    for batch in test_loader:
        if isinstance(batch, (tuple, list)):
            src_batch, tgt_batch = batch[0], batch[1]
        else:
            src_batch = tgt_batch = batch
        src_batch = src_batch.to(device)
        tgt_batch = tgt_batch.to(device)
        def safe_sum(tensor):
            if tensor.dim() == 1:
                return [int((tensor != 0).sum().item())]
            return (tensor != 0).sum(dim=1).tolist()
        src_lens = safe_sum(src_batch)
        tgt_lens = safe_sum(tgt_batch)
        for i in range(len(src_lens)):
            if collected >= n_samples:
                break
            src_len = src_lens[i]
            tgt_len = tgt_lens[i]
            if src_batch.dim() == 2:
                latin_idxs = src_batch[i, :src_len].tolist()
                tgt_seq = tgt_batch[i, 1:tgt_len-1].tolist()
            else:
                latin_idxs = src_batch[:src_len].tolist()
                tgt_seq = tgt_batch[1:tgt_len-1].tolist()
            latin_str = ''.join([source_vocab_inv.get(idx, '?') for idx in latin_idxs])
            true_str = ''.join([inv_target_vocab.get(idx, '?') for idx in tgt_seq])
            pred_idxs = predict_single(model, latin_idxs, dataset, device)
            pred_str = ''.join([inv_target_vocab.get(idx, '?') for idx in pred_idxs])
            data_rows.append([latin_str, true_str, pred_str, "Yes" if pred_str == true_str else "No"])
            collected += 1
        if collected >= n_samples:
            break

    # Now create and log the table ONCE
    table = wandb.Table(data=data_rows, columns=["Word", "Translation", "Prediction", "Correct"])
    wandb.log({"Test Predictions": table})
    return table

