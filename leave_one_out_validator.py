import numpy as np

class LeaveOneOutValidator:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def validate(self, classifier, features):
        correct = 0
        for i in range(len(self.data)):
            train_data = np.delete(self.data, i, axis=0)
            train_labels = np.delete(self.labels, i)
            test_data = self.data[i]
            test_label = self.labels[i]

            classifier.fit(train_data[:, features], train_labels)
            predicted_label = classifier.predict(test_data[features].reshape(1, -1))

            if predicted_label == test_label:
                correct += 1

        accuracy = correct / len(self.data)
        return accuracy
