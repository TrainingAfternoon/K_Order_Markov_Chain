'''
Accreditations:
Developed from the code provided here: https://www.educative.io/blog/deep-learning-text-generation-markov-chains
beemovie.txt taken from: https://gist.githubusercontent.com/The5heepDev/a15539b297a7862af4f12ce07fee6bb7/raw/7164813a9b8d0a3b2dcffd5b80005f1967887475/entire_bee_movie_script
huckfinn.txt taken from: https://www.gutenberg.org/ebooks/76
bukowy.txt taken from: https://www.researchgate.net/publication/318341911_Increased_Perfusion_Pressure_Drives_Renal_T-Cell_Infiltration_in_the_Dahl_Salt-Sensitive_Rat

Revisions and code organization done by: Sam Keyser 11/6/2021
CS3400 Machine Learning Final Project
'''
import numpy as np
import re

class KOrderMarkovModel:

    def __init__(self, k=4):
        self.k = k
        self.T = None

    def fit(self, filepath):
        self.T = self._convert_freq_into_prob(self._generate_table(self._load_text(filepath)))

    def predict(self, sentence_fragment, max_len=10000) -> str:
        if len(sentence_fragment) < self.k:
            raise Exception('Length of the sentence fragment must be at least the length of the context, k')

        if self.T is None:
            raise Exception('Model has not been fit yet')

        return self._generate_text(sentence_fragment, max_len)

    def _generate_table(self, data):
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

    def _convert_freq_into_prob(self, T):
        for kx in T.keys():
            s = float(sum(T[kx].values()))
            for k in T[kx].keys():
                T[kx][k] = T[kx][k] / s

        return T

    def _load_text(self, filename):

        with open(filename, encoding='utf8') as f:
            return f.read().lower().replace('\n', ' ')

    def _sample_next(self, ctx, model):

        ctx = ctx[-self.k:]
        if model.get(ctx) is None:
            return " "
        possible_Chars = list(model[ctx].keys())
        possible_values = list(model[ctx].values())

        return np.random.choice(possible_Chars, p=possible_values)

    def _generate_text(self, starting_sent, maxLen):

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
    k = int(input('K = '))
    corpus = input('Training File: ')
    sentence_fragment = input('Sentence: ')
    prediction_len = int(input('Prediction output length: '))

    model = KOrderMarkovModel(k)
    model.fit(corpus)

    print('\n'.join([sub for sub in split(model.predict(sentence_fragment, prediction_len), 84)]))

if __name__ == '__main__':
    main()
