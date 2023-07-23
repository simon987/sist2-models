import json
from clip.simple_tokenizer import SimpleTokenizer

if __name__ == "__main__":

    tok = SimpleTokenizer()
    encoder = tok.encoder

    ranks = dict(("`".join(k), v) for k, v in tok.bpe_ranks.items())

    with open("models/tokenizer.json", "w") as f:
        json.dump({
            "bpe_ranks": ranks,
            "encoder": encoder
        }, f)
