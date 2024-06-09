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
        last_best_accuracy = 0

        for i in range(num_features):
            current_best_feature = None
            current_best_accuracy = 0

            for feature in range(num_features):
                if feature in best_features:
                    continue

                accuracy = self.validator.validate(self.classifier, best_features + [feature])
                print(f"Using feature(s) {[f+1 for f in best_features] + [feature+1]} accuracy is {accuracy:.2f}")

                if accuracy > current_best_accuracy:
                    current_best_accuracy = accuracy
                    current_best_feature = feature

            if current_best_accuracy - last_best_accuracy < min_accuracy_improvement:
                print("Accuracy has not improved. Stopping.")
                break

            last_best_accuracy = current_best_accuracy

            if current_best_accuracy > best_accuracy:
                best_accuracy = current_best_accuracy
                best_features.append(current_best_feature)

            print(f"Feature set {[f+1 for f in best_features]} was best, accuracy is {best_accuracy:.2f}\n")

        print(f"\nBest feature subset (forward selection): {[f+1 for f in best_features]} with accuracy: {best_accuracy:.2f}\n")

    def backwardElimination(self):
        num_features = self.validator.data.shape[1]
        best_features = list(range(num_features))
        best_accuracy = self.validator.validate(self.classifier, best_features)
        print(f"\nStarting with feature set {[f+1 for f in best_features]} with accuracy {best_accuracy:.2f}")

        while len(best_features) > 1:
            current_best_feature_set = best_features
            current_best_accuracy = best_accuracy
            improvement = False

            for feature in best_features:
                feature_set = [f for f in best_features if f != feature]
                accuracy = self.validator.validate(self.classifier, feature_set)
                print(f"Using feature(s) {[f+1 for f in feature_set]} accuracy is {accuracy:.2f}")

                if accuracy > current_best_accuracy:
                    current_best_accuracy = accuracy
                    current_best_feature_set = feature_set
                    improvement = True

            if improvement:
                best_accuracy = current_best_accuracy
                best_features = current_best_feature_set
                print(f"Feature set {[f+1 for f in best_features]} was best, accuracy is {best_accuracy:.2f}\n")
            else:
                break

        print(f"\nBest feature subset (backward elimination): {[f+1 for f in best_features]} with accuracy: {best_accuracy:.2f}")

    def SCVforwardSelection(self):
        best_features = []
        remaining_features = list(range(self.validator.data.shape[1]))
        best_accuracy = 0
        last_best_accuracy = 0
        min_accuracy_improvement = 0.001

        while remaining_features:
            best_feature = None
            current_best_accuracy = 0

            for feature in remaining_features:
                current_features = best_features + [feature]
                accuracy = self.validator.stratified_cross_validation(self.classifier, current_features)
                print(f"Using feature(s) {[f+1 for f in best_features] + [feature+1]} accuracy is {accuracy:.2f}")

                if accuracy > current_best_accuracy:
                    current_best_accuracy = accuracy
                    best_feature = feature

            if current_best_accuracy - last_best_accuracy < min_accuracy_improvement:
                print("Accuracy has not improved. Stopping.")
                break

            last_best_accuracy = current_best_accuracy

            if best_feature is not None:
                best_features.append(best_feature)
                remaining_features.remove(best_feature)
                print(f"Feature set {[f+1 for f in best_features]} was best, accuracy is {current_best_accuracy:.2f}\n")
            else:
                break

        print(f"\nBest feature subset (SCV forward selection): {[f+1 for f in best_features]} with accuracy: {current_best_accuracy:.2f}")

    def SCVbackwardElimination(self):
        best_features = list(range(self.validator.data.shape[1]))
        best_accuracy = self.validator.stratified_cross_validation(self.classifier, best_features)
        print(f"\nStarting with feature set {[f+1 for f in best_features]} with accuracy {best_accuracy:.2f}")

        while len(best_features) > 1:
            current_best_feature_set = best_features
            current_best_accuracy = best_accuracy
            improvement = False

            for feature in best_features:
                current_features = [f for f in best_features if f != feature]
                accuracy = self.validator.stratified_cross_validation(self.classifier, current_features)
                print(f"Using feature(s) {[f+1 for f in current_features]} accuracy is {accuracy:.2f}")

                if accuracy > current_best_accuracy:
                    current_best_accuracy = accuracy
                    current_best_feature_set = current_features
                    improvement = True

            if improvement:
                best_accuracy = current_best_accuracy
                best_features = current_best_feature_set
                print(f"Feature set {[f+1 for f in best_features]} was best, accuracy is {best_accuracy:.2f}\n")
            else:
                break

        print(f"\nBest feature subset (SCV backward elimination): {[f+1 for f in best_features]} with accuracy: {best_accuracy:.2f}")

    def BobAlgo(self, n_components=5):
        print("\nRunning Bob the Builder's Special Algo (SVD)...")
        svd = TruncatedSVD(n_components=n_components)
        svd.fit(self.validator.data)
        top_features = np.argsort(-svd.components_[0])[:n_components]
        print(f"Top features selected: {[f+1 for f in top_features]}")
        accuracy = self.validator.stratified_cross_validation(self.classifier, top_features)
        print(f"Accuracy with selected features: {accuracy:.2f}\n")