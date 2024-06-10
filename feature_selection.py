import random
from sklearn.decomposition import TruncatedSVD
import numpy as np


class FeatureSelection:
    def __init__(self, validator, classifier):
        self.validator = validator
        self.classifier = classifier

    def forwardSelection(self, min_accuracy_improvement=0.001):
        best_features = []
        best_accuracy = 0
        num_features = self.validator.data.shape[1]
        highest_accuracy = best_accuracy
        highest_accuracy_features = best_features.copy()

        while len(best_features) < num_features:
            current_best_feature = None
            current_best_accuracy = -1  # Initialize to a value less than possible accuracy

            for feature in range(num_features):
                if feature in best_features:
                    continue

                feature_set = best_features + [feature]
                accuracy = self.validator.validate(self.classifier, feature_set)
                print(f"Using feature(s) {[f+1 for f in feature_set]} accuracy is {accuracy:.2f}")

                if accuracy > current_best_accuracy:
                    current_best_accuracy = accuracy
                    current_best_feature = feature

            if current_best_feature is not None:
                best_features.append(current_best_feature)
                best_accuracy = current_best_accuracy
                print(f"Feature set {[f+1 for f in best_features]} was best, accuracy is {best_accuracy:.2f}\n")

                if best_accuracy > highest_accuracy:
                    highest_accuracy = best_accuracy
                    highest_accuracy_features = best_features.copy()

        print(f"\nBest feature subset (forward selection): {[f+1 for f in highest_accuracy_features]} with accuracy: {highest_accuracy:.2f}\n")


    def backwardElimination(self):
        num_features = self.validator.data.shape[1]
        best_features = list(range(num_features))
        best_accuracy = self.validator.validate(self.classifier, best_features)
        print(f"\nStarting with feature set {[f+1 for f in best_features]} with accuracy {best_accuracy:.2f}")

        highest_accuracy = best_accuracy
        highest_accuracy_features = best_features.copy()

        while len(best_features) > 1:
            current_best_accuracy = -1
            current_worst_feature = None

            for feature in best_features:
                feature_set = [f for f in best_features if f != feature]
                accuracy = self.validator.validate(self.classifier, feature_set)
                print(f"Using feature(s) {[f+1 for f in feature_set]} accuracy is {accuracy:.2f}")

                if accuracy > current_best_accuracy:
                    current_best_accuracy = accuracy
                    current_worst_feature = feature

            if current_worst_feature is not None:
                best_features.remove(current_worst_feature)
                best_accuracy = current_best_accuracy
                print(f"Feature set {[f+1 for f in best_features]} was best, accuracy is {best_accuracy:.2f}\n")

                if best_accuracy > highest_accuracy:
                    highest_accuracy = best_accuracy
                    highest_accuracy_features = best_features.copy()

        print(f"\nBest feature subset (backward elimination): {[f+1 for f in highest_accuracy_features]} with accuracy: {highest_accuracy:.2f}")

    def SCVforwardSelection(self):
        best_features = []
        remaining_features = list(range(self.validator.data.shape[1]))
        highest_accuracy = 0
        highest_accuracy_features = best_features.copy()
        min_accuracy_improvement = 0.001

        while remaining_features:
            current_best_feature = None
            current_best_accuracy = -1  # Initialize to a value less than possible accuracy

            for feature in remaining_features:
                feature_set = best_features + [feature]
                accuracy = self.validator.stratified_cross_validation(self.classifier, feature_set)
                print(f"Using feature(s) {[f+1 for f in feature_set]} accuracy is {accuracy:.2f}")

                if accuracy > current_best_accuracy:
                    current_best_accuracy = accuracy
                    current_best_feature = feature

            if current_best_feature is not None:
                best_features.append(current_best_feature)
                remaining_features.remove(current_best_feature)
                print(f"Feature set {[f+1 for f in best_features]} was best, accuracy is {current_best_accuracy:.2f}\n")

                if current_best_accuracy - highest_accuracy >= min_accuracy_improvement:
                    highest_accuracy = current_best_accuracy
                    highest_accuracy_features = best_features.copy()
            else:
                break

        print(f"\nBest feature subset (SCV forward selection): {[f+1 for f in highest_accuracy_features]} with accuracy: {highest_accuracy:.2f}\n")


    def SCVbackwardElimination(self):
        num_features = self.validator.data.shape[1]
        best_features = list(range(num_features))
        best_accuracy = self.validator.stratified_cross_validation(self.classifier, best_features)
        print(f"\nStarting with feature set {[f+1 for f in best_features]} with accuracy {best_accuracy:.2f}")

        highest_accuracy = best_accuracy
        highest_accuracy_features = best_features.copy()

        while len(best_features) > 1:
            current_best_accuracy = -1
            current_worst_feature = None

            for feature in best_features:
                feature_set = [f for f in best_features if f != feature]
                accuracy = self.validator.stratified_cross_validation(self.classifier, feature_set)
                print(f"Using feature(s) {[f+1 for f in feature_set]} accuracy is {accuracy:.2f}")

                if accuracy > current_best_accuracy:
                    current_best_accuracy = accuracy
                    current_worst_feature = feature

            if current_worst_feature is not None:
                best_features.remove(current_worst_feature)
                best_accuracy = current_best_accuracy
                print(f"Feature set {[f+1 for f in best_features]} was best, accuracy is {best_accuracy:.2f}\n")

                if best_accuracy > highest_accuracy:
                    highest_accuracy = best_accuracy
                    highest_accuracy_features = best_features.copy()

        print(f"\nBest feature subset (SCV backward elimination): {[f+1 for f in highest_accuracy_features]} with accuracy: {highest_accuracy:.2f}")



    def BobAlgo(self, n_components=5):
        print("\nRunning Bob the Builder's Special Algo (SVD)...")
        svd = TruncatedSVD(n_components=n_components)
        svd.fit(self.validator.data)
        top_features = np.argsort(-svd.components_[0])[:n_components]
        print(f"Top features selected: {[f+1 for f in top_features]}")
        accuracy = self.validator.stratified_cross_validation(self.classifier, top_features)
        print(f"Accuracy with selected features: {accuracy:.2f}\n")
