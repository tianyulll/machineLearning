import numpy as np
from sklearn.preprocessing import normalize
import sys
# import textwrap

symbol_to_index = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

# get the decoded state path
def viterbi_decode(obs, states, start_p, trans_p, emit_p):

    T = len(obs)
    N = len(states)
    
    # Initialize tables
    viterbi = np.zeros((T, N))
    backpointers = np.zeros((T, N), dtype=int)
    
    # Initialize first column
    viterbi[0] = np.log(start_p * emit_p[:,obs[0]])
    # calculate matrix
    for t in range(1, T):
        for s in range(N):
            prev_probs = viterbi[t-1] + np.log(trans_p[:,s] * emit_p[s,obs[t]])
            viterbi[t,s] = np.max(prev_probs) 
            backpointers[t,s] = np.argmax(prev_probs)
    
    # Backtrack
    path = [np.argmax(viterbi[-1])]
    # print("table", viterbi)
    for t in range(T-1, 0, -1):
        path.append(backpointers[t,path[-1]])
    path.reverse()
    
    return path

# re-estimate HMM parameters with MLE
def mle(seqs, paths, states, transPrev, emitPrev):
    
    #count hidden states in the decoded path
    count_hidden = np.ones(len(states))
    for path in paths:
        for state in path:
            count_hidden[state] += 1
    # print("init count", count_hidden)
    for i, seq in enumerate(seqs):
        # initialize count matrix with pseudoount
        trans_count = np.zeros_like(transPrev)+1e-9
        emit_count = np.zeros_like(emitPrev)+1e-9
        path = paths[i]
        for i in range(1, len(path)):
            prev_state = path[i-1]
            curr_state = path[i]
            trans_count[prev_state, curr_state] += 1
            emit_count[curr_state, seq[i]] += 1

    # normalize everything:
    initial_P = normalize(count_hidden.reshape(1,-1), axis=1, norm='l1')
    trans_P = normalize(trans_count, axis=1, norm='l1')
    emit_P = normalize(emit_count, axis=1, norm='l1')

    return initial_P, trans_P, emit_P

# train HMM model with EM
def train(seqs, states, initialProb, transitProb, emitProb, conv=0.001, maxIter=100):
    
    for iter in range(maxIter):
        decodePath = []
        if iter%10==0: print("in iteration", iter) 
        # generated decoded path
        for seq in seqs:
            decode=viterbi_decode(seq, states, initialProb, transitProb, emitProb)
            decodePath.append(decode)
        # re-estimate parameter
        newInitProb, newTransitProb, newEmitProb = mle(seqs, decodePath, states, transitProb, emitProb)
        if np.allclose(newInitProb, initialProb, atol=conv) and \
              np.allclose(newTransitProb, transitProb, atol=conv) and \
                np.allclose(newEmitProb, emitProb, atol=conv):
            print("early converge with iteration", iter)
            break
        initialProb = newInitProb
        transitProb = newTransitProb
        emitProb = newEmitProb
    return initialProb, transitProb, emitProb

def parseFasta(fileName):
    # read file
    seqs = [] 
    with open(fileName, 'r') as f:
        seq_id = None
        seq = ''
        for line in f:
            if line.startswith('>'):
                if seq_id is not None:
                    seqs.append(seq)
                seq_id = line.strip()[1:]
                seq = ''
            else:
                seq += line.strip()
        if seq_id is not None:
            seqs.append(seq)
    # encode sequences
    encoded_seq = [[symbol_to_index[s] for s in seq] for seq in seqs]
    return encoded_seq


if __name__ == "__main__":

    trainSeq = parseFasta(sys.argv[1])
    #print("data", trainSeq[0])
    hmm_states=[0,1] #vampire=0, werewolf=1
    initial_prob=np.array([0.5,0.5])
    trans_init = np.array([[0.75, 0.25], [0.25, 0.75]])
    emit_init = np.array([[0.3, 0.2, 0.2, 0.3], [0.2, 0.3, 0.3, 0.2]]) # in order ACGT

    finalInit, finalTrans, finalEmit = train(trainSeq, hmm_states, initial_prob, trans_init, emit_init, conv=0.0001)
    print("infered Pi:", '\n',finalInit)
    print("infered transition:",'\n', finalTrans)
    print("infered emission:", '\n',finalEmit)

    testSeq = parseFasta(sys.argv[2])
        
    decodedPath = viterbi_decode(testSeq[0], hmm_states, finalInit, finalTrans, finalEmit)
    decodedState=''
    for i in decodedPath:
        if i == 0: 
            decodedState+='V' 
        else:
            decodedState+='W' 
    with open('inferedParaPath.txt', 'w') as f:
        # seperated = textwrap.fill(decodedState, 60)
        f.write(decodedState)
    f.close()

    trueTrans = np.array([[0.99, 0.01], [0.014, 0.986]])
    trueEmit = np.array([[0.35, 0.15, 0.15, 0.35], [0.175, 0.325, 0.325, 0.175]])
    labeledTest = viterbi_decode(testSeq[0], hmm_states, initial_prob, trueTrans, trueEmit)
    decodedState=''
    for i in labeledTest:
        if i == 0: 
            decodedState+='V' 
        else:
            decodedState+='W' 
    with open('trueParaPath2.txt', 'w') as f:
        # seperated = textwrap.fill(decodedState, 60)
        f.write(decodedState)
    f.close()

 