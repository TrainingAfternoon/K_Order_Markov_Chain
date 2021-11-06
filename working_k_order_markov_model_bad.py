'''
Developed from the code provided here: TODO
'''
import numpy as np

def generateTable(data,k=4):
    
    T = {}
    for i in range(len(data)-k):
        X = data[i:i+k]
        Y = data[i+k]
        
        if T.get(X) is None:
            T[X] = {}
            T[X][Y] = 1
        else:
            if T[X].get(Y) is None:
                T[X][Y] = 1
            else:
                T[X][Y] += 1
    
    return T


def convertFreqIntoProb(T):
    for kx in T.keys():
        s = float(sum(T[kx].values()))
        for k in T[kx].keys():
            T[kx][k] = T[kx][k]/s

    return T


def load_text(filename):
    with open(filename,encoding='utf8') as f:
        return f.read().lower()


def MarkovChain(text,k=4):
    T = generateTable(text,k)
    T = convertFreqIntoProb(T)
    return T


def sample_next(ctx,model,k):

    ctx = ctx[-k:]
    if model.get(ctx) is None:
        return " "
    possible_Chars = list(model[ctx].keys())
    possible_values = list(model[ctx].values())

    print(possible_Chars)
    print(possible_values)

    return np.random.choice(possible_Chars,p=possible_values)

def generateText(model,starting_sent,k=4,maxLen=1000):

    sentence = starting_sent
    ctx = starting_sent[-k:]

    for ix in range(maxLen):
        next_prediction = sample_next(ctx,model,k)
        sentence += next_prediction
        ctx = sentence[-k:]
    return sentence


def main():
    text_data_fp = input('filepath for training: ')
    chain = MarkovChain(load_text(text_data_fp), int(input('k: ')))
    print(generateText(chain,input('sentence fragment: ')))






if __name__ == '__main__':
    main()
