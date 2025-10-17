import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)

model_name = "luerhard/PopBERT"

# 1) Load tokenizer & model (keep 4 labels, multi-label)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)
model.config.problem_type = "multi_label_classification"  # ensures BCEWithLogitsLoss
model.config.id2label = {
    0: "anti_elitism",
    1: "people_centrism",
    2: "left_wing",
    3: "right_wing",
}
model.config.label2id = {v: k for k, v in model.config.id2label.items()}

# 2) Tiny toy dataset (each label is 0/1; multi-label possible)
texts = [
    "Das ist Klassenkampf von oben.",
    "Wir, das Volk, werden übergangen!",
    "Solidarität mit den Arbeitnehmerinnen und Arbeitnehmern.",
    "Die Elite entfernt sich vom Volk.",
]
labels = [
    [1, 0, 1, 0],  # anti-elitism + left
    [0, 1, 0, 1],  # people-centrism + right
    [0, 0, 1, 0],  # left only
    [1, 0, 0, 0],  # anti-elitism only
]
ds = Dataset.from_dict({"text": texts, "labels": labels})

def tokenize(batch):
    enc = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)
    enc["labels"] = batch["labels"]  # shape: (4,) floats/ints
    return enc

ds = ds.map(tokenize, batched=True, remove_columns=["text"])

# 3) Train (minimal settings)
args = TrainingArguments(
    output_dir="./popbert-ml",
    per_device_train_batch_size=2,
    num_train_epochs=2,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_steps=1,
    save_strategy="no",
)

trainer = Trainer(model=model, args=args, train_dataset=ds)
trainer.train()

# 4) Inference: sigmoid -> probabilities; optional thresholding

test_text = "Das ist Klassenkampf von oben, im Interesse der Besitzenden."
enc = tokenizer(test_text, return_tensors="pt", truncation=True)
with torch.inference_mode():
    logits = model(**enc).logits
probs = torch.sigmoid(logits).squeeze(0)

print("probs:", {model.config.id2label[i]: float(probs[i]) for i in range(4)})