from tqdm import tqdm
import numpy as np
print("Loading Glove 840B Word Vectors")
#embedding_file = '../glove.840B/glove.840B.300d.txt'
embedding_file = '../glove.twitter.27B/glove.twitter.27B.200d.txt'
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.split(" ")) for o in tqdm(open(embedding_file)))
#max_features=len(embeddings_index.values()[0])
max_features=len(list(embeddings_index.values())[0])
def get_all_embeddings():
  return embeddings_index
def get_vector(sentence):
    final_vector = np.zeros(max_features)
    total = 0
    if sentence is None or isinstance(sentence,str)==False:
        return final_vector

    words = sentence.split(' ')
    for word in words:
        vector = embeddings_index.get(word.lower())
        if vector is not None:
            total = total+1
            final_vector = final_vector + vector
        if total ==0:
            total=1 #To prevent div by 0
        final_vector = final_vector / total
    return final_vector

def get_vectors(sentences):
    final_vectors=[]
    for sentence in tqdm(sentences):
        vector = get_vector(sentence)
        final_vectors.append(vector)
    return final_vectors

def get_vector4(sentence):
  final_vector = np.zeros(max_features)
  total = 0
  words = sentence.split(' ')
  for word in words:
    vector = embeddings_index.get(word.lower())
    if vector is not None:
      total = total+1
      final_vector = final_vector + vector
  if total ==0:
    total=1 #To prevent div by 0
  final_vector = final_vector / total
  return final_vector

def get_vectors4(sentences):
  final_vectors=[]
  for sentence in tqdm(sentences):
    vector = get_vector(sentence)
    final_vectors.append(vector)
  return final_vectors

def get_vector2(sentence):
  final_vector = np.zeros(max_features)
  total = 0
  words = sentence.split(' ')
  for word in words:
    vector = embeddings_index.get(word.lower())
    if vector is not None:
      total = total+1
      final_vector = final_vector + vector
    else:
      print( word.lower())
  if total ==0:
    total=1 #To prevent div by 0
  final_vector = final_vector / total
  return final_vector

def get_most_similar(word, howmany):
  word_vector = get_vector(word)

  similarities = {}
  for w, v in tqdm(embeddings_index.items()):
    sim = word_vector.dot(v)
    similarities[w]=sim

  return sorted(similarities.items(), key=lambda x:x[1], reverse=True)[:howmany]


def get_most_similar_by_vector(word_vector, howmany):
  similarities = {}
  norm_word_vector = np.linalg.norm(word_vector)
  for w, v in tqdm(embeddings_index.items()):
    norm_v = np.linalg.norm(v)
    sim = word_vector.dot(v)/(norm_v*norm_word_vector)
    similarities[w]=sim

  return sorted(similarities.items(), key=lambda x:x[1], reverse=True)[:howmany]
