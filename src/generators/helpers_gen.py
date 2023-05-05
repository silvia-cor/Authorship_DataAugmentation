import torch
import torch.nn.functional as F


# PositionalEncoding expects shape [seq_len, batch_size, embedding_dim]
def forward_trans(trans_model, x, dense):  # [B, S, E]
    if dense:
        x = trans_model.dense_emb(x)  # [B, S, E]
    x = torch.transpose(x, 0, 1)  # [S, B, E]
    positional_output = trans_model.positional_encoder(x)  # [S, B, E]
    trans_output = trans_model.trans_encoder(positional_output)  # [S, B, E]
    logits = trans_output[-1, :, :]  # [1, E]
    output = F.relu(trans_model.dense(logits))  # [1, E]
    return torch.unsqueeze(output, dim=0)  # [1, 1, E]


def forward_gru(gru_model, x, dense):  # [B, S, E]
    if dense:
        x = gru_model.dense_emb(x)  # [B, S, E]
    gru_out, _ = gru_model.gru(x)  # [B, S, E]
    logits = gru_out[:, -1, :]  # [1, E]
    output = F.relu(gru_model.dense(logits))  # [1, E]
    return torch.unsqueeze(output, dim=0)  # [1, 1, E]


def get_examples_for_LM(texts, tokenizer):
    examples = []
    for text in texts:
        text = tokenizer.encode(text, return_tensors="pt")[0]
        for i in range(6, len(text) + 1):
            examples.append(text[:i])
    lm_inputs, lm_targets = [], []
    for ex in examples:
        lm_inputs.append(ex[:-1])
        lm_targets.append(ex[-1])
    return lm_inputs, lm_targets


def compute_loss(optimizer, loss, batch_size, i, n):
    loss = loss / batch_size
    loss.backward()
    if ((i + 1) % batch_size == 0) or (i + 1 == n):
        optimizer.step()
        optimizer.zero_grad()
    return loss.item(), optimizer
