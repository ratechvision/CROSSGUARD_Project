import json
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model


class TextEncoder(nn.Module):

    def __init__(
        self,
        model_name="distilbert-base-uncased",
        emb_dim=256
    ):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        backbone = AutoModel.from_pretrained(model_name)

        config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["q_lin", "v_lin"],
            bias="none"
        )

        self.backbone = get_peft_model(backbone, config)

        hidden = self.backbone.config.hidden_size

        self.project = nn.Sequential(
            nn.Linear(hidden, emb_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, texts):

        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        device = next(self.parameters()).device
        tokens = {k: v.to(device) for k, v in tokens.items()}

        output = self.backbone(**tokens)

        cls = output.last_hidden_state[:, 0]

        return self.project(cls)


def load_reports(folder="data/threat_reports"):

    reports = []

    for file in sorted(Path(folder).glob("threat_report_*.json")):

        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        text = data.get("description", "")

        if text == "":
            text = data.get("summary", "")

        reports.append(text)

    return reports


if __name__ == "__main__":

    texts = load_reports()

    model = TextEncoder()

    emb = model(texts)

    print("=" * 50)
    print("CROSSGUARD Text Encoder")
    print("=" * 50)
    print("Reports Loaded :", len(texts))
    print("Embedding Shape:", emb.shape)