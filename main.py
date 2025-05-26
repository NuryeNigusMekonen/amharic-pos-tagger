from src.train import train_model
from src.evaluate import evaluate_model
from src.utils import load_config

if __name__ == "__main__":
    config = load_config("config.yaml")
    model, vocab, tag2idx = train_model(config)
    evaluate_model(model, config, vocab, tag2idx)