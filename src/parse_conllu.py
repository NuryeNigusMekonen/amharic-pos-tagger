def parse_conllu(filepath, pos_col=3):
    sentences = []
    current = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith("#") or line.strip() == "":
                if current:
                    sentences.append(current)
                    current = []
                continue
            parts = line.strip().split("\t")
            if "-" in parts[0]:
                continue
            word, pos = parts[1], parts[pos_col]
            current.append((word, pos))
        if current:
            sentences.append(current)
    return sentences