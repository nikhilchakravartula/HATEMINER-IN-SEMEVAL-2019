import nltk
import torch
from models import InferSent

#First, download the InferSent pre-trained models by running this in the InferSent directory:
#curl -Lo encoder/infersent1.pkl https://s3.amazonaws.com/senteval/infersent/infersent1.pkl
#curl -Lo encoder/infersent2.pkl https://s3.amazonaws.com/senteval/infersent/infersent2.pkl
# Next, set the W2V_PATH variable below and clickbait sentences accordingly.
# output is numpy array of 4096 dim. space.

V=2
MODEL_PATH = '/data/InferSent/encoder/infersent%s.pkl' % V
params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
infersent = InferSent(params_model)
infersent.load_state_dict(torch.load(MODEL_PATH))

W2V_PATH = '/data/glove.840B/glove.840B.300d.txt'
#W2V_PATH = '/data/fasttext/crawl-300d-2M-subword.vec'
infersent.set_w2v_path(W2V_PATH)

def get_vectors(sentences):
  sentences = [s.lower() for s in sentences]
  #sentences = ["Hello, I am bakhtiyar", "wow here is a cake for you!"]
  infersent.build_vocab(sentences, tokenize=True)
  embeddings = infersent.encode(sentences, tokenize =  True)
  #print(embeddings)
  return embeddings
#print (embeddings)
#print(len(embeddings), len(embeddings[0]))
