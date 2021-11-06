'''
Developed from the code provided here: https://www.educative.io/blog/deep-learning-text-generation-markov-chains
Beemovie.txt taken from: https://gist.githubusercontent.com/The5heepDev/a15539b297a7862af4f12ce07fee6bb7/raw/7164813a9b8d0a3b2dcffd5b80005f1967887475/entire_bee_movie_script
'''
import numpy as np
import re

class KOrderMarkovModel:


    def __init__(self, k=4):
        self.k = k

    def fit(self, filepath):
        self.T = self._convertFreqIntoProb(self._generateTable(self._load_text(filepath)))

    def predict(self, sentence_fragment, max_len=10000) -> str:
        return self._generateText(sentence_fragment, max_len)

    def _generateTable(self, data):
        k = self.k
        T = {}
        for i in range(len(data) - k):
            X = data[i:i + k]
            Y = data[i + k]

            if T.get(X) is None:
                T[X] = {}
                T[X][Y] = 1
            else:
                if T[X].get(Y) is None:
                    T[X][Y] = 1
                else:
                    T[X][Y] += 1

        return T

    def _convertFreqIntoProb(self, T):
        for kx in T.keys():
            s = float(sum(T[kx].values()))
            for k in T[kx].keys():
                T[kx][k] = T[kx][k] / s

        return T

    def _load_text(self, filename):
        with open(filename, encoding='utf8') as f:
            #print(re.sub('\s+', ' ', ' '.join(f.readlines()).replace('\n', ' ').lower()))
            #return re.sub('\s+', ' ', ' '.join(f.readlines()).replace('\n', ' ').lower())
            return f.read().lower().replace('\n', ' ')

    def _sample_next(self, ctx, model):

        ctx = ctx[-self.k:]
        if model.get(ctx) is None:
            return " "
        possible_Chars = list(model[ctx].keys())
        possible_values = list(model[ctx].values())

        print(possible_Chars)
        print(possible_values)

        return np.random.choice(possible_Chars, p=possible_values)

    def _generateText(self, starting_sent, maxLen):

        sentence = starting_sent
        ctx = starting_sent[-self.k:]

        for ix in range(maxLen):
            next_prediction = self._sample_next(ctx, self.T)
            sentence += next_prediction
            ctx = sentence[-self.k:]
        return sentence

def split(str, slice):
    k = slice
    while k <= len(str):
        yield str[k - slice:k]
        k += slice


def main():
    sentence_fragment = 'dear'
    model = KOrderMarkovModel(len(sentence_fragment))
    model.fit('train_corpus.txt')
    print('\n'.join([sub for sub in split(model.predict(sentence_fragment, 256), 80)]))

if __name__ == '__main__':
    main()
