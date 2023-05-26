import argparse
import numpy as np

'''
usage:
python3 forwardbackward.py toy_data/validation.txt \
toy_data/index_to_word.txt toy_data/index_to_tag.txt \
toy_data/hmminit.txt toy_data/hmmemit.txt \
toy_data/hmmtrans.txt toy_data/predicted.txt \
toy_data/metrics.txt
Empirical_script:
for i in 10 100 1000 10000; do
echo $i ;
python3 forwardbackward.py en_data/train.txt \
en_data/index_to_word.txt en_data/index_to_tag.txt \
en_${i}_init.txt en_${i}_emit.txt \
en_${i}_trans.txt trial/en_${i}_train_predicted.txt \
trial/en_${i}_train_metrics.txt
python3 forwardbackward.py en_data/validation.txt \
en_data/index_to_word.txt en_data/index_to_tag.txt \
en_${i}_init.txt en_${i}_emit.txt \
en_${i}_trans.txt trial/en_${i}_valid_predicted.txt \
trial/en_${i}_valid_metrics.txt
done
'''

def get_inputs():
    """
    Collects all the inputs from the command line and returns the data. To use this function:

        validation_data, words_to_indices, tags_to_indices, hmminit, hmmemit, hmmtrans, predicted_file, metric_file = parse_args()

    Where above the arguments have the following types:

        validation_data --> A list of validation examples, where each element is a list:
            validation_data[i] = [(word1, tag1), (word2, tag2), (word3, tag3), ...]
        
        words_to_indices --> A dictionary mapping words to indices

        tags_to_indices --> A dictionary mapping tags to indices

        hmminit --> A np.ndarray matrix representing the initial probabilities

        hmmemit --> A np.ndarray matrix representing the emission probabilities

        hmmtrans --> A np.ndarray matrix representing the transition probabilities

        predicted_file --> A file path (string) to which you should write your predictions

        metric_file --> A file path (string) to which you should write your metrics
    """

    parser = argparse.ArgumentParser()
    
    parser.add_argument("validation_data", type=str)
    parser.add_argument("index_to_word", type=str)
    parser.add_argument("index_to_tag", type=str)
    parser.add_argument("hmminit", type=str)
    parser.add_argument("hmmemit", type=str)
    parser.add_argument("hmmtrans", type=str)
    parser.add_argument("predicted_file", type=str)
    parser.add_argument("metric_file", type=str)

    args = parser.parse_args()

    validation_data = list()
    with open(args.validation_data, "r") as f:
        examples = f.read().strip().split("\n\n")
        for example in examples:
            xi = [pair.split("\t") for pair in example.split("\n")]
            validation_data.append(xi)
    
    with open(args.index_to_word, "r") as g:
        words_to_indices = {w: i for i, w in enumerate(g.read().strip().split("\n"))}
    
    with open(args.index_to_tag, "r") as h:
        tags_to_indices = {t: i for i, t in enumerate(h.read().strip().split("\n"))}
    
    hmminit = np.loadtxt(args.hmminit, dtype=float, delimiter=" ")
    hmmemit = np.loadtxt(args.hmmemit, dtype=float, delimiter=" ")
    hmmtrans = np.loadtxt(args.hmmtrans, dtype=float, delimiter=" ")

    return validation_data, words_to_indices, tags_to_indices, hmminit, hmmemit, hmmtrans, args.predicted_file, args.metric_file

# You should implement a logsumexp function that takes in either a vector or matrix
# and performs the log-sum-exp trick on the vector, or on the rows of the matrix
def logsumexp(x, vector=False):
    xMax = np.max(x)
    # if vector
    if vector: return xMax + np.log(np.sum(np.exp(x - xMax)))
    return xMax + np.log(np.sum(np.exp(x - xMax), axis=1))


def forwardbackward(seq, loginit, logtrans, logemit, words_to_indices, tags_to_indices):
    """
    Your implementation of the forward-backward algorithm.

        seq is an input sequence, a list of words (represented as strings)

        loginit is a np.ndarray matrix containing the log of the initial matrix

        logtrans is a np.ndarray matrix containing the log of the transition matrix

        logemit is a np.ndarray matrix containing the log of the emission matrix

        words_to_indices --> A dictionary mapping words to indices

        tags_to_indices --> A dictionary mapping tags to indices

    You should compute the log-alpha and log-beta values and predict the tags for this sequence.
    """
    L = len(seq)
    M = len(loginit)

    encodedSeq = [words_to_indices[w] for w in seq]
    indices_to_tags = {v: k for k, v in tags_to_indices.items()} # get reversed dictionary

    # Initialize log_alpha and fill it in - feel free to use words_to_indices to index the specific word
    alpha = np.zeros(shape=(L, M))
    alpha[0,:] = loginit + logemit[:, encodedSeq[0]]
    for i in range(1, len(alpha)):
        expPart = alpha[i-1,:]+logtrans.T
        alpha[i, :] = logemit[:, encodedSeq[i]] + logsumexp(expPart)

    # Initialize log_beta and fill it in - feel free to use words_to_indices to index the specific word
    beta = np.zeros(shape=(L, M))
    for i in range(L-2, -1, -1):
        # beta=A(i+1)b(i+1)B
        expPart = logemit[:, encodedSeq[i+1]] + beta[i+1,:] + logtrans
        beta[i,:] = logsumexp(expPart)

    # Compute the predicted tags for the sequence - tags_to_indices can be used to index to the rwquired tag
    predict = []
    for i in range(L):
        prob = alpha[i] + beta[i]
        predict.append(indices_to_tags[np.argmax(prob)])
    # Compute the stable log-probability of the sequence

    # Return the predicted tags and the log-probability
    return predict, logsumexp(alpha[-1], vector=True)
    

    
    
if __name__ == "__main__":
    # Get the input data
    validation_data, words_to_indices, tags_to_indices, hmminit, hmmemit, hmmtrans, predicted_file, metric_file = get_inputs()

    # For each sequence, run forward_backward to get the predicted tags and 
    # the log-probability of that sequence.
    ll = []
    count, totalTag = 0, 0
    with open(predicted_file, 'w+') as f:
        for data in validation_data:
            seq = [datai[0] for datai in data]
            correctStates = [datai[1] for datai in data]
            predict, prob = forwardbackward(seq, np.log(hmminit), np.log(hmmtrans), np.log(hmmemit), words_to_indices, tags_to_indices)
            ll.append(prob)
            # count accuracy
            totalTag+=len(correctStates)
            for i in range(len(seq)):
                if correctStates[i] == predict[i]: count+=1
            # write output file
            for i in range(len(seq)):
                print(seq[i]+'\t'+predict[i], file=f)
            print(file=f)
    f.close()

    # Compute the average log-likelihood and the accuracy. The average log-likelihood 
    # is just the average of the log-likelihood over all sequences. The accuracy is 
    # the total number of correct tags across all sequences divided by the total number 
    # of tags across all sequences.
    with open(metric_file, 'w+') as f:
        print("Average Log-Likelihood:", np.mean(ll), file=f)
        print("Accuracy:", count/totalTag, file=f)
    f.close()
