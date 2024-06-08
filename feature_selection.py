class FeatureSelection:
    def __init__(self, validator, classifier):
        self.validator = validator
        self.classifier = classifier
    
    
    def forward_selection(self, min_accuracy_improvement=0.001):
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

        print(f"\nBest feature subset (forward selection): {[f+1 for f in best_features]} with accuracy: {best_accuracy:.2f}\n\n")



    
    def backward_elimination(self):
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


