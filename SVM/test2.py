import numpy as np

def convert_to_dense_if_sparse(X):
    if hasattr(X, 'toarray'):
        return X.toarray()
    return X

class SVM_classifier():
    def __init__(self, learing_rate=0.01, lambda_=100, iterations=1000):
        self.learning_rate = learing_rate
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
                di = W
                dA = 0
            else:
                di = W - (self.lambda_ * Y_batch[idx] * X_batch[idx])
                dA = 0 - (self.lambda_ * Y_batch[idx])
            dw += di
            dB += dA
        dw = dw / len(Y_batch)  # average
        dB = dB / len(Y_batch)  # avg
        return dw, dB
        
    def fit(self, features, outputs) -> bool:
        # print(features)
        # features = convert_to_dense_if_sparse(features)

        # print(features.shape)
        max_epochs = self.iterations
        weights = np.zeros(features.shape[1])
        bias = 0
        nth = 0

        prev_cost = float("inf")
        cost_threshold = 0.0001  # in percent
        
        for epoch in range(1, max_epochs):
            gradW, gradB = self.gradientDescent(weights, bias, features, outputs)
            weights = weights - (self.learning_rate * gradW)
            bias = bias - (self.learning_rate * gradB)

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
            
        self.W = weights
        self.B = bias
        return True
    
    def decisionFunc(self, X):
        # X = convert_to_dense_if_sparse(X)
        ans = []
        for x in X:
            ans.append(np.dot(x, self.W) + self.B)
        return np.array(ans)
    
    def predict(self, X):
        # X = convert_to_dense_if_sparse(X)
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
        

class OvRClassifier:
    def __init__(self, n_classes, learning_rate=0.01, lambda_param=10, n_iters=1000):
        self.n_classes = n_classes
        self.models = [SVM_classifier(learning_rate, lambda_param, n_iters) for _ in range(n_classes)]

    def fit(self, X, y):
        for i in range(self.n_classes):
            y_binary = np.where(y == i, 1, -1)  # Convert current class to 1 and others to -1
            self.models[i].fit(X, y_binary)

    def predict(self, X):
        decision_values = np.array([model.decisionFunc(X) for model in self.models])
        return np.argmax(decision_values, axis=0)
    
    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        correct = 0
        cnt_pos = 0
        cnt_neg = 0
        cnt_net = 0
        for i in range(predictions.shape[0]):
            if predictions[i] == y_test[i]:
                correct += 1
            
            if y_test[i] == 1:
                cnt_pos += 1
            elif y_test[i] == -1:
                cnt_neg += 1
            else:
                cnt_neg += 1
                
        accuracy = correct / y_test.shape[0]
        print(f'Accuracy: {accuracy * 100:.2f}%')
        print(f"Pos_rate: {cnt_pos}, {cnt_neg}, {cnt_net}")
        return accuracy


# Example usage
if __name__ == "__main__":
    X_train = np.array([[1, 2], [2, 3], [3, 3], [2, 1], [3, 2], [7, 8], [8, 9], [9, 8], [7, 7], [9, 9], [4, 1], [5, 2]])
    y_train = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2])

    # Initialize the One-vs-Rest classifier for 3 classes
    classifier = OvRClassifier(n_classes=3)
    classifier.fit(X_train, y_train)

    # Predict on training data
    predictions = classifier.predict(X_train)
    print("Predictions:", predictions)
    print("True labels:", y_train)

    classifier.evaluate(X_train, y_train)