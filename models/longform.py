import torch
import torch.nn as nn
from transformers import LongformerModel

# Must match the class definition used during training
class LongformerTripleClassifier(nn.Module):
    def __init__(self, model_name, dropout=0.1):
        super().__init__()
        self.longformer = LongformerModel.from_pretrained(model_name)
        hidden_size     = self.longformer.config.hidden_size
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, input_ids, attention_mask, global_attention_mask=None):
        out = self.longformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
        )
        hidden = out.last_hidden_state
        mask   = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return type('Output', (), {'logits': logits})()


def build_model(weights_path="models/longformer_triple_best.pt"):
    model = LongformerTripleClassifier("allenai/longformer-base-4096").float()
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model