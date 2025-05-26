#evaluation metrics
import torch
from sklearn.metrics import classification_report

def evaluate_model(model, config, vocab, tag2idx):
    model.eval()
    sample = ["እሱ", "አመታት", "አለ"]
    idxs = torch.tensor([[vocab.get(w, vocab["<UNK>"]) for w in sample]])
    with torch.no_grad():
        out = model(idxs)
        preds = out.argmax(-1).squeeze().tolist()

    inv_tag = {v: k for k, v in tag2idx.items()}
    print("Prediction:")
    for word, pred in zip(sample, preds):
        print(f"{word} → {inv_tag[pred]}")


def evaluate_on_dataset(model, dataloader, tag2idx):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x, y in dataloader:
            out = model(x)
            preds = out.argmax(-1)
            for p, t in zip(preds, y):
                for pi, ti in zip(p, t):
                    if ti.item() != -100:
                        all_preds.append(pi.item())
                        all_labels.append(ti.item())

    inv_tag = {v: k for k, v in tag2idx.items()}
    pred_tags = [inv_tag[i] for i in all_preds]
    true_tags = [inv_tag[i] for i in all_labels]
    print("\nClassification Report:")
    print(classification_report(true_tags, pred_tags, digits=4))