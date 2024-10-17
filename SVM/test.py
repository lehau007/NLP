import numpy as np

def convert_to_dense_if_sparse(X):
    if hasattr(X, 'toarray'):
        return X.toarray()
    return X

class SVM_classifier():
    def __init__(self, learning_rate=0.01, lambda_=10, iterations=1000):
        self.learning_rate = learning_rate
        self.lambda_ = lambda_
        self.iterations = iterations
        self.W = None
        self.B = None
    
    def compute_hingle_loss(self, W, B, X_batch, y_batch):
        """ calculate hinge loss """
        N = X_batch.shape[0]
        distance = []
        for idx, x in enumerate(X_batch):
            distance.append(max(0, 1 - y_batch[idx] * (np.dot(x, W) + B)))

        distances = np.array(distance) # let distance into numpy array 
    
        hinge_loss = self.lambda_ * (np.sum(distances) / N) # find hinge loss
        
        # calculate cost
        cost = 1 / 2 * np.dot(W, W) + hinge_loss
        return cost

    def gradientDescent(self, W, B, X_batch, Y_batch):
        distance = []
        for idx, x in enumerate(X_batch):
            distance.append(1 - Y_batch[idx] * (np.dot(x, W) + B))
    
        dw = np.zeros(len(W))
        dB = 0
        for idx, d in enumerate(distance):
            if max(0, d) == 0:
                dw += W
                dB += 0
            else:
                dw += W - (self.lambda_ * Y_batch[idx] * X_batch[idx])
                dB += 0 - (self.lambda_ * Y_batch[idx])
        
        dw = dw / len(Y_batch)  # average
        dB = dB / len(Y_batch)  # avg
        return dw, dB
        
    def fit(self, features, outputs) -> bool:
        # print(features)
        features = convert_to_dense_if_sparse(features)

        # print(features.shape)
        max_epochs = self.iterations
        weights = np.zeros(features.shape[1])
        bias = 0
        nth = 0

        prev_cost = float("inf")
        cost_threshold = 0.01  # in percent
        
        for epoch in range(1, max_epochs):
            gradW, gradB = self.gradientDescent(weights, bias, features, outputs)

            # convergence check on 2^nth epoch
            if epoch == 2 ** nth or epoch == max_epochs - 1:
                cost = self.compute_hingle_loss(weights, bias, features, outputs)
                print("Epoch is:{} and Cost is: {}".format(epoch, cost))
                # stoppage criterion
                if abs(prev_cost - cost) < cost_threshold * prev_cost:
                    self.W = weights
                    self.B = bias
                    return True
                prev_cost = cost
                nth += 1
            
            # update grad
            weights = weights - (self.learning_rate * gradW)
            bias = bias - (self.learning_rate * gradB)
            
        self.W = weights
        self.B = bias
        return True
    
    def decisionFunc(self, X):
        X = convert_to_dense_if_sparse(X)
        ans = []
        for x in X:
            ans.append(np.dot(x, self.W) + self.B)
        return np.array(ans)
    
    def predict(self, X):
        X = convert_to_dense_if_sparse(X)
        # print(X)
        prediction = []
        for x in X:
            prediction.append(np.dot(x, self.W) + self.B) # w.x + b
        
        # print(np.sign(prediction))
        return np.sign(prediction)

    # Evaluate the model
    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        correct = 0
        cnt_pos = 0
        cnt_neg = 0
        for i in range(predictions.shape[0]):
            if predictions[i] == y_test[i]:
                correct += 1
            
            if y_test[i] == 1:
                cnt_pos += 1
            else:
                cnt_neg += 1
        accuracy = correct / y_test.shape[0]
        print(f'Accuracy: {accuracy * 100:.2f}%')
        print(f"Pos_rate: {cnt_pos / y_test.shape[0] * 100:.2}%")
        return accuracy
def polynomial_kernel(x, y, degree=3, coef=1):
    return (np.dot(x, y) + coef) ** degree

class SVM:
    def __init__(self, learning_rate=0.01, lambda_param=0.1, n_iters=1000, kernel=polynomial_kernel):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.kernel = kernel
        self.alpha = None
        self.b = 0

    def fit(self, X, y_):
        # y is a np array just have -1, 1 value
        n_samples, n_features = X.shape
        self.alpha = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            
            # Gradient Descent
            for i in range(n_samples):
                condition = y_[i] * self.decision_function(X[i], X, y_) >= 1
                if condition:
                    dw = 0
                    db = 0
                else:
                    for j in range(n_features):
                        dw = -y_[i] * self.kernel(X[i], X[j])
                        self.alpha[j] = self.alpha[j] * self.lambda_param - self.lr * dw

                    db = -y_[i]
                    self.b -= self.lr * db

    def decision_function(self, x, X_train, y_train):
        decision_value = 0
        for i in range(len(X_train)):
            decision_value += self.alpha[i] * y_train[i] * self.kernel(X_train[i], x)
        return decision_value + self.b

    def predict(self, X):
        predictions = []
        for x in X:
            prediction = np.sign(self.decision_function(x, X_train, y_train))
            predictions.append(prediction)
        return np.array(predictions)

# Example usage
if __name__ == "__main__":
    # Create a small dataset
    X_train = np.array([[3, 0], [2, 1], [1, 2], [2, 3], [3, 3]])
    y_train = np.array([1, 1, 1, -1, -1])

    # Initialize and train the kernel SVM
    clf = SVM(learning_rate=0.01, lambda_param=0.1, n_iters=1000, kernel=polynomial_kernel)
    clf.fit(X_train, y_train)

    # Predict on the training data
    predictions = clf.predict(X_train)
    print("Predictions:", predictions)
    print("True labels:", y_train)

    clf = SVM_classifier(learning_rate=0.01, lambda_=10, iterations=1000)
    clf.fit(X_train, y_train)

    # Predict on the training data
    predictions = clf.predict(X_train)
    print("Predictions:", predictions)
    print("True labels:", y_train)
