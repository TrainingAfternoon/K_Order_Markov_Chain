# K_Order_Markov_Chain
Final Presentation project for CS 3400, Machine Learning, completed Fall 2021 @ the Milwaukee School of Engineering.

Based on the work done by Ryan Thelin of [educative.io](www.educative.io), which may be found [here](https://www.educative.io/blog/deep-learning-text-generation-markov-chains). Reorganization of the code done by Samuel Keyser.

Thanks to [gutenberg](www.gutenberg.org) for providing raw text files, which were used to train this model.

Project completed in conjunction with Jack Haek and Alexander Blake.

--------------------------------

K-Order Markov Chains are a simple predictive text generation model, much like the kind found in any cellphone nowadays, based on the foundational work done by the Russian mathematician Andrey Markov. Markov created Markov chains to analyze the distribution of vowel-vowel, consonant-vowel, and consonant-consonant pairings in *Eugene Onegin*.


Markov chains are a mathematical tool that allow for the prediction of a future state of a system, based on the current state. Markov chains obey the markov property, so they do not rely on information about previous states; they only consider the current state.


Markov chains are organized as *Transition Matrices*. Consider a three-state system, A, B, C. Transition matrices are right stochastic matrices, so each row is a probability vector, which represents the probability that the state will transition from a given state to another state. Rows represent current states, columns represent potential states.


+---+----+----+----+
| . | A  | B  | C  |
+---+----+----+----+
| A | .6 | .3 | .1 |
| - |  - |  - |  - |
| B | .5 | .4 | .1 |
| - |  - |  - |  - |
| C | .3 | .6 | .1 |
+---+----+----+----+


