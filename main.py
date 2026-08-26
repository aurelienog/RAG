import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('qwen3..')

sentences = ["blabla", "blablabla", "bla"]

embeddings = model.encode(sentences)

