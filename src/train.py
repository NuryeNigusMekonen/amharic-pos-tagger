import torch
from torch.utils.data import DataLoader
from src.model import BiLSTMTagger
from src.preprocess import build_vocab, build_tagset
from src.dataset import POSDataset, collate_fn
from src.parse_conllu import parse_conllu
from src.evaluate import evaluate_on_dataset

def train_model(config):
    device = torch.device("cuda" if config['use_gpu'] and torch.cuda.is_available() else "cpu")

    sentences = parse_conllu(config['conllu_file'])
    vocab = build_vocab(sentences)
    tag2idx = build_tagset(sentences)
    dataset = POSDataset(sentences, vocab, tag2idx)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)

    model = BiLSTMTagger(len(vocab), len(tag2idx), config['embedding_dim'], config['hidden_dim']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

    for epoch in range(config['epochs']):
        model.train()
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out.view(-1, out.shape[-1]), y.view(-1))
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    evaluate_on_dataset(model, dataloader, tag2idx)
    torch.save(model.state_dict(), "bilstm_model.pt")
    return model, vocab, tag2idx