def predict_single(model, src_idx, dataset, device, max_len=50):
    """Greedy decode using attention-based decoder"""
    model.eval()
    with torch.no_grad():
        src_tensor = torch.tensor(src_idx, dtype=torch.long).unsqueeze(0).to(device)
        src_lens = [len(src_idx)]
        
        # Fixed encoder call
        encoder_outputs, hidden = model.encoder(
            src_tensor, 
            torch.tensor(src_lens, dtype=torch.long).cpu()
        )
        
        # Fixed mask creation
        src_mask = model.create_mask(src_tensor)
        
        input_char = torch.tensor([dataset.target_vocab['<sos>']], device=device)
        output_idxs = []

        for _ in range(max_len):
            logits, hidden = model.decoder(input_char, hidden, encoder_outputs, src_mask)
            top1 = logits.argmax(1).item()

            if top1 == dataset.target_vocab['<eos>']:
                break

            output_idxs.append(top1)
            input_char = torch.tensor([top1], device=device)

    return output_idxs

def log_test_predictions(model, dataset, test_loader, device, n_samples=20):
    """Logs test predictions with dimension handling"""
    import wandb
    table = wandb.Table(columns=["Word", "Translation", "Prediction", "Correct"])
    
    source_vocab_inv = {v: k for k, v in dataset.source_vocab.items()}
    inv_target_vocab = {v: k for k, v in dataset.target_vocab.items()}
    
    collected = 0
    for batch in test_loader:
        # Handle different batch formats
        if len(batch) == 2:
            src_batch, tgt_batch = batch
        else:
            src_batch, tgt_batch = batch, batch
            
        src_batch = src_batch.to(device)
        tgt_batch = tgt_batch.to(device)
        
        # Fixed length calculation
        def safe_sum(tensor):
            if tensor.dim() == 1:  # Handle 1D tensors
                return [int((tensor != 0).sum().item())]
            return (tensor != 0).sum(dim=1).tolist()
            
        src_lens = safe_sum(src_batch)
        tgt_lens = safe_sum(tgt_batch)

        for i in range(len(src_lens)):
            if collected >= n_samples:
                break
                
            # Get actual sequence lengths
            src_len = src_lens[i]
            tgt_len = tgt_lens[i]
            
            # Handle 1D vs 2D tensors
            if src_batch.dim() == 2:
                latin_idxs = src_batch[i, :src_len].tolist()
                tgt_seq = tgt_batch[i, 1:tgt_len-1].tolist()  # Exclude SOS/EOS
            else:  # 1D tensor
                latin_idxs = src_batch[:src_len].tolist()
                tgt_seq = tgt_batch[1:tgt_len-1].tolist()
            
            # Convert to strings
            latin_str = ''.join([source_vocab_inv.get(idx, '?') for idx in latin_idxs])
            true_str = ''.join([inv_target_vocab.get(idx, '?') for idx in tgt_seq])
            
            # Prediction
            pred_idxs = predict_single(model, latin_idxs, dataset, device)
            pred_str = ''.join([inv_target_vocab.get(idx, '?') for idx in pred_idxs])
            
            table.add_data(latin_str, true_str, pred_str, "Yes" if pred_str == true_str else "No")
            collected += 1

    wandb.log({"Test Predictions with attention" : table})
    return table

