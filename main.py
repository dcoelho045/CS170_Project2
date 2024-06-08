import numpy as np
from feature_selection import FeatureSelection
from leave_one_out_validator import LeaveOneOutValidator
from knn_classifier import KNNClassifier

def load_dataset(file_path):
    data = np.loadtxt(file_path)
    labels = data[:, 0]  # the first column is the label
    features = data[:, 1:]  # The remaining columns are the features
    return features, labels

def main():
    print("Welcome to Bob the Builder's Feature Selection Algorithm! by jgonz671, abrem005, and dcoel003.")
    print("Select the dataset file to use:")
    print("1. small-test-dataset-1.txt")
    print("2. large-test-dataset-1.txt")
    print("3. CS170_Spring_2024_Small_data__39.txt")
    print("4. CS170_Spring_2024_Large_data__39.txt")
    file_path = input("Enter your choice (1, 2, 3, or 4): ")
    
    file_map = {
        '1': 'small-test-dataset-1.txt',
        '2': 'large-test-dataset-1.txt',
        '3': 'CS170_Spring_2024_Small_data__39.txt',
        '4': 'CS170_Spring_2024_Large_data__39.txt'
    }
    file_path = file_map[file_path]

    data, labels = load_dataset(file_path)
    validator = LeaveOneOutValidator(data, labels)
    classifier = KNNClassifier()

    feature_selection = FeatureSelection(validator, classifier)
    
    print("\nType the number of the algorithm you want to run.")
    print("1. Forward Selection")
    print("2. Backward Elimination")
    algorithm_choice = input("Enter your choice of algorithm: 1, 2, or 3: ")


    if algorithm_choice == '1':
        print("\nRunning Forward Selection...")
        feature_selection.forward_selection()
    elif algorithm_choice == '2':
        print("\nRunning Backward Elimination...")
        feature_selection.backward_elimination()
    # Add cases for extra credit algorithms

if __name__ == "__main__":
    main()
