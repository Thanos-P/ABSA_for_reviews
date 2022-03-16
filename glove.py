import numpy as np
import myUtils
import nltk


def prepare_embedding_retrieval(glove_file):
    """
        Read a GloVe embeddings file and return a matrix with the vocab_size first words
        and two dictionaries, one mapping words to indices and one mapping indices to words.
    """
    vocab = {}
    embeddings = []

    with open(glove_file, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file.readlines()):
            items = line.strip().split(' ')
            if items[0] == '<unk>':
                continue

            vocab[items[0]] = i
            embeddings.append([float(x) for x in items[1:]])

    emb_matrix = np.array(embeddings)
    del embeddings

    # normalize each word vector
    emb_matrix /= np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    return emb_matrix, vocab


class GloveModel:
    """A glove model with a similarity query function"""

    def __init__(self, emb_norm=None, vocab=None):
        self.emb_norm = emb_norm
        self.vocab = vocab
        if emb_norm is not None and vocab is not None:
            self.initialized = True
        else:
            self.initialized = False

    def __getitem__(self, item):
        """Get embedding from a word or an id"""
        if not self.initialized:
            raise ValueError('Glove model is not initialized')

        try:
            if type(item) is str:
                return self.emb_norm[self.vocab[item]]
            elif type(item) is int or type(item) is np.int32:
                return self.emb_norm[item]
            else:
                raise ValueError('Accepted types for query embedding are "int" and "str"')
        except KeyError:
            print(f'Word "{item}" doesn\'t exist in the vocabulary')
            return None
        except IndexError:
            print(f'Id "{item}" doesn\'t exist in the vocabulary')
            return None

    @classmethod
    def from_pretrained(cls, glove_file):
        emb_norm, vocab = prepare_embedding_retrieval(glove_file)
        return cls(emb_norm, vocab)

    def get_id(self, token):
        """Get id from a word"""
        if not self.initialized:
            raise ValueError('Glove model is not initialized')

        try:
            return self.vocab[token]
        except KeyError:
            # print(f'Word "{token}" doesn\'t exist in the vocabulary')
            return None

    def string_to_ids(self, text, return_none=False):
        """Get token ids from a string"""
        text = myUtils.pad_punctuation(myUtils.remove_accents(myUtils.remove_numbers(text.lower())))
        tokens = nltk.word_tokenize(text)
        ids = []
        for token in tokens:
            token_id = self.get_id(token)
            if return_none or token_id is not None:
                ids.append(token_id)
        return ids

    def string_to_embeddings(self, text):
        """Get token embeddings from a string"""
        text = myUtils.pad_punctuation(myUtils.remove_accents(myUtils.remove_numbers(text.lower())))
        tokens = nltk.word_tokenize(text)
        embeddings = []
        for token in tokens:
            token_emb = self[token]
            if token_emb is not None:
                embeddings.append(token_emb)
        return embeddings

    @myUtils.cache(pos=(1, 2))
    def most_similar(self, query_word, k=1, return_similarities=False):
        """Find k most similar words to query word using cosine similarity"""

        # Find query word embedding
        query_emb = self[query_word]
        if query_emb is None:
            return None

        # compute pairwise similarities
        similarities = {}
        for word, word_id in self.vocab.items():
            if not word == query_word:
                similarities[word] = np.dot(query_emb, self.emb_norm[word_id])

        if return_similarities:
            result = sorted(similarities.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[0:k]
        else:
            result = []
            temp = sorted(similarities.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
            for key, val in temp[0:k]:
                result.append(key)

        return result
