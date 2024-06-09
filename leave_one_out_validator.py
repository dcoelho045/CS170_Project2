import numpy as np
from sklearn.model_selection import StratifiedKFold

class LeaveOneOutValidator:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def validate(self, classifier, features): # Leave-one-out cross validation
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
    
    # This method is used to evaluate the accuracy of a classifier using stratified cross validation
    def _evaluate_accuracy(self, classifier, test_data, test_labels, features):
        correct_predictions = 0
        num_instances = len(test_data)

        for i in range(num_instances):
            test_instance = test_data[i, features]
            predicted_label = classifier.predict(test_instance.reshape(1, -1))
            if predicted_label == test_labels[i]:
                correct_predictions += 1

        accuracy = correct_predictions / num_instances
        return accuracy

    # Stratified cross validation
    def stratified_cross_validation(self, classifier, features, num_folds=5): 
        skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
        accuracies = []

        for train_index, test_index in skf.split(self.data, self.labels):
            training_data, test_data = self.data[train_index], self.data[test_index]
            training_labels, test_labels = self.labels[train_index], self.labels[test_index]

            classifier.fit(training_data[:, features], training_labels)
            fold_accuracy = self._evaluate_accuracy(classifier, test_data, test_labels, features)
            accuracies.append(fold_accuracy)

        mean_accuracy = np.mean(accuracies)
        return mean_accuracy