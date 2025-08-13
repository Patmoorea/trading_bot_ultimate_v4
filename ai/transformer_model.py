from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

class TransformerModel:
    def __init__(self, model_name="bert-base-uncased", device="cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = device
        self.model.to(self.device)

    def initialize(self):
        pass # Optionnel

    def predict(self, features):
        # features = dict ou texte, à adapter selon ton use case
        # Si tu passes des indicateurs en texte :
        text = " ".join([f"{k}:{v:.3f}" for k, v in features.items()])
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        score = torch.softmax(logits, dim=1)[0][1].item()  # Proba "long"
        return score