from data import *
from nearest_neighbor import *

def nearest_neighbor(data):
    accuracies = []
    print("The accuracy of nearest neighbor, standardized nearest neighbor, and normalized nearest neighbor are as follows:\n")
    pred = nearest(data.X_train, data.y_train, data.X_test, data.y_test)  # Accessing attributes through the instance
    print("Training and testing data loaded successfully for basic data")
    accuracies.append(accuracy(pred, data.y_test))
    print(f'\tThe accuracy of nearest neighbor {accuracies[0]:.3f}\n')
    std_pred = nearest(data.X_train_std, data.y_train, data.X_test_std, data.y_test)  # Accessing attributes through the instance
    print("Training and testing data loaded successfully for standardized data")
    accuracies.append(accuracy(std_pred, data.y_test))
    print(f'\tThe accuracy of standardized nearest neighbor {accuracies[1]:.3f}\n')
    norm_pred = nearest(data.X_train_norm, data.y_train, data.X_test_norm, data.y_test)  # Accessing attributes through the instance
    print("Training and testing data loaded successfully for normalized data")
    accuracies.append(accuracy(norm_pred, data.y_test))
    print(f'\tThe accuracy of normalized nearest neighbor {accuracies[2]:.3f}\n')
    std_norm_pred = nearest(data.X_train_std_norm, data.y_train, data.X_test_std_norm, data.y_test)  # Accessing attributes through the instance
    print("Training and testing data loaded successfully for normalized and standardized data")
    accuracies.append(accuracy(std_norm_pred, data.y_test))
    print(f'\tThe accuracy of normalized and standardized nearest neighbor {accuracies[3]:.3f}\n')
    
    largest_accuracy = max(accuracies)
    index_of_largest_accuracy = accuracies.index(largest_accuracy)
    model = ["unmodified", "standardized", "normalized", "normalized and standardized"]
    
    print(f'The nearest neighbor model with the largest accuracy has {model[index_of_largest_accuracy]} data with an accuracy of {largest_accuracy*100:.5f}%\n')
    
    return largest_accuracy

def compare(data):
    overall_accuracies = []
    for i in range(10):
        overall_accuracies.append(nearest_neighbor(data))
        data = Data('training.csv', 'testing.csv')#Reload the data to randomize it more
    overall_accuracies = np.array(overall_accuracies)
    print(f'\nThe average accuracy of the nearest neighbor model is {np.mean(overall_accuracies)*100:.5f}% with a standard deviation of {np.std(overall_accuracies)*100:.5f}%\n')

if __name__ == '__main__':
    data = Data('training.csv', 'testing.csv')
    #compare(data)  #Uncomment this line to compare the accuracies of the nearest neighbor model to get an average accuracy and standard deviation over multiple runs
    nearest_neighbor(data)