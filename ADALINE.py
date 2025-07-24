import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


inputs = np.array([  # Input patterns
    [1, 1],
    [1, 2],
    [2, -1],
    [2, 0],
    [-1, 2],
    [-2, 1],
    [-1, -1],
    [-2, -2]
])

targets = np.array([ # Target values 
    [-1, -1],
    [-1, -1],
    [-1, 1],
    [-1, 1],
    [1, -1],
    [1, -1],
    [1, 1],
    [1, 1]
])


W = np.array([[1.0, 0.0], [0.0, 1.0]]) # Initialize weights 
b = np.array([[1.0], [1.0]])           # Initialize biases
alpha = 0.04
max_iterations = 50 # maximum number of iterations


def purelin(x): # Activation function
    return x

# Training function
def train_adaline(inputs, targets, W, b, alpha, max_iterations):
    converged = False
    iteration = 0
    while not converged and iteration < max_iterations:
        converged = True
        for i in range(len(inputs)):
            p = inputs[i].reshape(2, 1)
            t = targets[i].reshape(2, 1)
            a = np.dot(W, p) + b
            e = t - a

           
            W = W + 2 * alpha * np.dot(e, p.T)  # Update weights
            b = b + 2 * alpha * e  # Update biases

            
            if iteration == 0 and i == 0:
                print("")
                print("For Test The Code")
                print("-------------------------")
                print(f"Updated Weights (W(1)): {W}")
                print(f"Updated Biases (b(1)): {b}")

            
            if iteration == 0 and i == 1:
                print(f"Updated Weights (W(2)): {W}")
                print(f"Updated Biases (b(2)): {b}")
                print("-------------------------")
                print("")


            if np.any(e != 0):
                converged = False
        iteration += 1
    return W, b, iteration, converged

# Train the model
W, b, iterations, converged = train_adaline(inputs, targets, W, b, alpha, max_iterations)
if converged:
    print(f"Training completed in {iterations} iterations.")
else:
    print(f"Training did not converge within {max_iterations} iterations.")

# Final weights and biases
print("")
print("Final weights and biases")
print("-------------------------")
print("Final weights:")
print(W)
print("Final biases:")
print(b)
print("-------------------------")
print("")


def test_adaline(inputs, targets, W, b): # Prepare data for the table
    print("                     Test Table")
    print("-------------------------------------------------------")
    data = {
        "Input": [list(p) for p in inputs],
        "Target": [list(t) for t in targets],
        "Output": [],
        "Error": [],
    }
    
    for i in range(len(inputs)):
        p = inputs[i].reshape(2, 1)
        t = targets[i].reshape(2, 1)
        a = purelin(np.dot(W, p) + b)
        e = t - a
        data["Output"].append([round(val, 3) for val in a.T.tolist()[0]])
        data["Error"].append([round(val, 3) for val in e.T.tolist()[0]])
    
    
    df = pd.DataFrame(data)
    print(df)


test_adaline(inputs, targets, W, b) # Call the function

#####
new_inputs = np.array([ #  4 new test vectors 
    [-1, -1],
    [-1, -2],
    [-2, 1],
    [-2, 2]
])


new_targets = np.array([
    [1, 1],
    [1, 1],
    [1, -1],
    [1, -1]
])

# Test the ADALINE with the new inputs and targets 
print("")
print("")
print("For New Input")
print("-------------------------------------------------------")
test_adaline(new_inputs, new_targets, W, b)  




#################################################### Plot decision boundaries
def plot_decision_boundaries(inputs, targets, W, b):
    plt.figure()

    # Plot the input points
    for i in range(len(inputs)):
        if targets[i][0] == 1 and targets[i][1] == 1:
            plt.scatter(inputs[i, 0], inputs[i, 1], marker='o', color='blue', label="Class 1" if i == 0 else "")
        elif targets[i][0] == -1 and targets[i][1] == -1:
            plt.scatter(inputs[i, 0], inputs[i, 1], marker='x', color='red', label="Class 3" if i == 0 else "")
        elif targets[i][0] == -1 and targets[i][1] == 1:
            plt.scatter(inputs[i, 0], inputs[i, 1], marker='s', color='green', label="Class 4" if i == 0 else "")
        else:
            plt.scatter(inputs[i, 0], inputs[i, 1], marker='^', color='black', label="Class 2" if i == 0 else "")

    
    x_min, x_max = inputs[:, 0].min() - 1, inputs[:, 0].max() + 1  # grid for decision boundaries
    y_min, y_max = inputs[:, 1].min() - 1, inputs[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    
    Z1 = purelin(np.dot(W[0:1], np.c_[xx.ravel(), yy.ravel()].T) + b[0]) # Compute decision boundaries for both outputs
    Z2 = purelin(np.dot(W[1:2], np.c_[xx.ravel(), yy.ravel()].T) + b[1])

    Z1 = Z1.reshape(xx.shape)
    Z2 = Z2.reshape(xx.shape)

    
    contour1 = plt.contour(xx, yy, Z1, levels=[0], colors='green', linestyles='-.') # Plot the decision boundaries
    contour2 = plt.contour(xx, yy, Z2, levels=[0], colors='orange', linestyles='-.')
    
    
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='green', linestyle='-.', label='Boundary 1'), # Add custom legend for contours
        Line2D([0], [0], color='orange', linestyle='-.', label='Boundary 2'),
    ]
    plt.legend(handles=legend_elements, loc='best')

    
    plt.axhline(0, color='black', linewidth=1.2) # Plot the axes at the origin
    plt.axvline(0, color='black', linewidth=1.2)
    plt.grid(True, which='both')

    plt.title("Decision Boundaries")
    plt.xlabel("Input 1 -- (P1 ) ")
    plt.ylabel("Input 2 -- (P2 )")
    plt.show()


# Call the updated function
plot_decision_boundaries(inputs, targets, W, b)


