#text preprocessing functions 
def build_vocab(sentences, min_freq=1):
    from collections import Counter
    word_freq = Counter(word for s in sentences for word, _ in s)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, freq in word_freq.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab

def build_tagset(sentences):
    tag_set = set(tag for s in sentences for _, tag in s)
    return {tag: idx for idx, tag in enumerate(sorted(tag_set))}

def encode_sentence(sentence, vocab, tag2idx):
    x = [vocab.get(w, vocab["<UNK>"]) for w, _ in sentence]
    y = [tag2idx[t] for _, t in sentence]
    return x, y