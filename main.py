import numpy as np
import time 
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

    yayNay = input("Would you like to normalize the data (y/n): ")    
    if(yayNay == 'y'):
        # Normalize the data
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        if np.any(std == 0):
            print("Warning: Some features have zero standard deviation.")
        std[std == 0] = 1
        data = (data - mean) / std
        print("Normalizing data . . . complete")

    validator = LeaveOneOutValidator(data, labels)
    classifier = KNNClassifier()
 

    yayNay = input("Would you like to stratified cross validation on the data (y/n): ")
    if(yayNay == 'y'):
        # allow for stratified cross validation function to be called here 
        features = list(range(data.shape[1]))  # Use all features initially
        validator_SCV = validator.stratified_cross_validation(classifier, features)
        feature_selection = FeatureSelection(validator_SCV, classifier) 
        print("Did stratified cross validation ")
    else:
        feature_selection = FeatureSelection(validator, classifier)
        
    
    print("\nType the number of the algorithm you want to run.")
    print("1. Forward Selection")
    print("2. Backward Elimination")
    print("3. Bob the Builder's Special Algo")
    algorithm_choice = input("Enter your choice of algorithm: 1, 2, or 3: ")


    if algorithm_choice == '1':
        print("\nRunning Forward Selection...")
        start_time = time.time()
        feature_selection.forwardSelection()
        end_time = time.time()
        print(f"Forward Selection took {end_time - start_time:.2f} seconds.")
    elif algorithm_choice == '2':
        print("\nRunning Backward Elimination...")
        start_time = time.time()
        feature_selection.backwardElimination()
        end_time = time.time()
        print(f"Backward Elimination took {end_time - start_time:.2f} seconds.") 
           
if __name__ == "__main__":
    main()
