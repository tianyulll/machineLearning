import numpy as np
import argparse
import csv
import matplotlib.pyplot as plt

def sigmoid(x : np.ndarray):
    """
    Implementation of the sigmoid function.

    Parameters:
        x (str): Input np.ndarray.

    Returns:
        An np.ndarray after applying the sigmoid function element-wise to the
        input.
    """
    e = np.exp(x)
    return e / (1 + e)


def loadData(file):
    x, y = [], []
    with open(file, encoding='utf-8') as f:
        read_file = csv.reader(f, delimiter='\t')
        for row in read_file:
            label, feature = row[0], row[1:]
            feature.append(1)
            x.append(np.array(feature, dtype=float))
            y.append(int(float(label)))
    return np.array(x), np.array(y)

# return the gradient
def gradient(theta, x, y):
    thetaX = np.matmul(theta.T, x.T)
    deltaJ = (sigmoid(thetaX) - y)*x
    return deltaJ

def train(
    theta : np.ndarray, # shape (D,) where D is feature dim
    X : np.ndarray,     # shape (N, D) where N is num of examples
    y : np.ndarray,     # shape (N,)
    num_epoch : int, 
    learning_rate : float
) -> None:

    theta = np.zeros(301)
    for epoch in range(num_epoch):
        for i, x in enumerate(X):
            deltaJ = gradient(theta, x, y[i])
            theta -= learning_rate*deltaJ
    return theta 

#negative log likelihood
def nll(theta, x, y):
    sigmoidRes = sigmoid(np.matmul(theta.T, x.T))
    return -y*np.log(sigmoidRes)-(1-y)*np.log(1-sigmoidRes)

# train and calculate loss
def trainLarge(theta, X, y, valX, valy, num_epoch, lr):
    theta = np.zeros(301)
    trainLoss, valLoss = [], []
    for epoch in range(num_epoch):
        if epoch % 100 == 0:
            print("in Epoch", epoch)
        trainE, valE = 0, 0
        # train and calculat loss
        for i, x in enumerate(X):
            deltaJ = gradient(theta, x, y[i])
            theta -= lr*deltaJ
        for i, x in enumerate(X):
            trainE += nll(theta, x, y[i])
        # loss on validation
        for i, valx in enumerate(valX):
            valE += nll(theta, valx, valy[i])
        # average negative log likelihodd
        trainLoss.append(trainE/len(X))
        valLoss.append(valE/len(valX))
    return theta, trainLoss, valLoss


def predict(
    theta : np.ndarray,
    X : np.ndarray
) -> np.ndarray:

    prediction = []
    for x in X:
        res = np.matmul(theta.T, x.T)
        if res >= 0: prediction.append(1.0)
        else: prediction.append(0.0)
    return np.array(prediction)

# compute error rate
def compute_error(
    y_pred : np.ndarray, 
    y : np.ndarray
) -> float:
    error = 0
    for i, yi in enumerate(y):
        if y_pred[i] != yi:
            error +=1
    return error/len(y)


def logisticRegression(trainFile, num, lr, trainout, testFile, testout, metric, valFile):
    # load train and validation dataset
    X, y = loadData(trainFile)
    valX, valy = loadData(valFile)

    # theta = train(np.zeros(301), X, y, num, lr)
    theta, trainLoss, valLoss = trainLarge(np.zeros(301), X, y, valX, valy, num, lr)

    # predict label for training
    pred = predict(theta, X)
    trainError = compute_error(pred, y)
    with open(trainout, 'w+') as f:
        for i in pred:
            f.write(str(int(i))+"\n")
    f.close()
    
    # predict label for test
    X, y = loadData(testFile)
    pred = predict(theta, X)
    testError = compute_error(pred, y)
    with open(testout, 'w+') as f:
        for i in pred:
            f.write(str(int(i))+"\n")
    f.close()
    
    # write the metrics
    with open(metric, 'w+') as f:
        print("error(train):", '%.6f' %trainError, file=f)
        print("error(test):", '%.6f' %testError, file=f)
    f.close()

    # #Generate plot
    # plt.plot(trainLoss, label = "train average NLL")
    # plt.plot(valLoss, label = "validate average NLL")
    # plt.xlabel("Epoch")
    # plt.ylabel("NLL")
    # plt.legend()
    # plt.savefig("train_performance.png")

    _, trainLoss2, _ = trainLarge(np.zeros(301), X, y, valX, valy, num, 0.01)
    _, trainLoss3, _ = trainLarge(np.zeros(301), X, y, valX, valy, num, 0.001)
    plt.plot(trainLoss, label = "lr=0.1")
    plt.plot(trainLoss2, label = "lr=0.01")
    plt.plot(trainLoss3, label = "lr=0.001")
    plt.xlabel("Epoch")
    plt.ylabel("NLL")
    plt.legend()
    plt.savefig("diff_lr_performance.png")




if __name__ == '__main__':
    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    # For example, to get the learning rate, you can use `args.learning_rate`.
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help='path to formatted training data')
    parser.add_argument("validation_input", type=str, help='path to formatted validation data')
    parser.add_argument("test_input", type=str, help='path to formatted test data')
    parser.add_argument("train_out", type=str, help='file to write train predictions to')
    parser.add_argument("test_out", type=str, help='file to write test predictions to')
    parser.add_argument("metrics_out", type=str, help='file to write metrics to')
    parser.add_argument("num_epoch", type=str, 
                        help='number of epochs of gradient descent to run')
    parser.add_argument("learning_rate", type=str, 
                        help='learning rate for gradient descent')
    args = parser.parse_args()

    logisticRegression(args.train_input, int(args.num_epoch), float(args.learning_rate), \
        args.train_out, args.test_input, args.test_out, args.metrics_out, args.validation_input)