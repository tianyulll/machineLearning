import argparse
import numpy as np

'''
usage
python3 learnhmm.py en_data/train.txt en_data/index_to_word.txt \
en_data/index_to_tag.txt en_init en_emit en_trans
'''

def get_inputs():
    """
    Collects all the inputs from the command line and returns the data. To use this function:

        train_data, words_to_index, tags_to_index, init_out, emit_out, trans_out = get_inputs()
    
    Where above the arguments have the following types:

        train_data --> A list of training examples, where each training example is a list
            of tuples train_data[i] = [(word1, tag1), (word2, tag2), (word3, tag3), ...]
        
        words_to_indices --> A dictionary mapping words to indices

        tags_to_indices --> A dictionary mapping tags to indices

        init_out --> A file path to which you should write your initial probabilities

        emit_out --> A file path to which you should write your emission probabilities

        trans_out --> A file path to which you should write your transition probabilities
    
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str)
    parser.add_argument("index_to_word", type=str)
    parser.add_argument("index_to_tag", type=str)
    parser.add_argument("hmmprior", type=str)
    parser.add_argument("hmmemit", type=str)
    parser.add_argument("hmmtrans", type=str)

    args = parser.parse_args()

    train_data = list()
    with open(args.train_input, "r") as f:
        examples = f.read().strip().split("\n\n")
        for example in examples:
            xi = [pair.split("\t") for pair in example.split("\n")]
            train_data.append(xi)
    
    with open(args.index_to_word, "r") as g:
        words_to_indices = {w: i for i, w in enumerate(g.read().strip().split("\n"))}
    
    with open(args.index_to_tag, "r") as h:
        tags_to_indices = {t: i for i, t in enumerate(h.read().strip().split("\n"))}
    
    return train_data, words_to_indices, tags_to_indices, args.hmmprior, args.hmmemit, args.hmmtrans

def saveParam(initial, emision, transition, fileName):
    # Add a pseudocount
    initial = initial+1
    emision = emision+1
    transition = transition+1

    # normalization
    initial = initial/sum(initial)
    emision = emision / emision.sum(axis=1, keepdims=True)
    transition = transition / transition.sum(axis=1, keepdims=True)
    
    # Save your matrices to the output files --- the reference solution uses 
    # np.savetxt (specify delimiter=" " for the matrices)
    np.savetxt(fileName+"_init.txt", initial, delimiter=" ")
    np.savetxt(fileName+"_emit.txt", emision, delimiter=" ")
    np.savetxt(fileName+"_trans.txt", transition, delimiter=" ")


if __name__ == "__main__":
    # Collect the input data
    train_data, words_to_index, tags_to_index, init_out, emit_out, trans_out = get_inputs()
    encodedData = [[(words_to_index[x[0]], tags_to_index[x[1]]) for x in datai] for datai in train_data]
    
    # Initialize the initial, emission, and transition matrices
    initial = np.zeros(shape=(len(tags_to_index),1))
    emision = np.zeros(shape=(len(tags_to_index), len(words_to_index)))
    transition = np.zeros(shape=(len(tags_to_index),len(tags_to_index)))
    
    # Increment the matrices
    count = 0
    for data in encodedData:
        initial[data[0][1]] += 1 #increment initial state
        for i, tup in enumerate(data):
            # tup[0] = word; tup[1] = tag
            emision[tup[1]][tup[0]] += 1 
            if i != 0 :
                transition[data[i-1][1]][tup[1]] += 1
        count += 1 
        if count == 10 or count == 100 or count == 1000 or count == 10000:
            saveParam(initial, emision, transition, "en_"+str(count))
        if count > 10000:
            print("finished")
            break

    '''
    # Increment the matrices
    for data in encodedData:
        initial[data[0][1]] += 1 #increment initial state
        for i, tup in enumerate(data):
            # tup[0] = word; tup[1] = tag
            emision[tup[1]][tup[0]] += 1 
            if i != 0 :
                transition[data[i-1][1]][tup[1]] += 1

    # Add a pseudocount
    initial = initial+1
    emision = emision+1
    transition = transition+1

    # normalization
    initial = initial/sum(initial)
    emision = emision / emision.sum(axis=1, keepdims=True)
    transition = transition / transition.sum(axis=1, keepdims=True)
    
    # Save your matrices to the output files --- the reference solution uses 
    # np.savetxt (specify delimiter=" " for the matrices)
    np.savetxt(init_out, initial, delimiter=" ")
    np.savetxt(emit_out, emision, delimiter=" ")
    np.savetxt(trans_out, transition, delimiter=" ")
    '''
