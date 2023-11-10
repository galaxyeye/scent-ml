import matplotlib
import numpy as np
import seaborn as sns
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd

matplotlib.rc('text', usetex=True)
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
sns.set_style("darkgrid")

import matplotlib.pyplot as plt

path = 'C:\\Users\\pereg\\AppData\\Local\\Temp\\pulsar\\amazon.dataset.csv'
samples = pd.read_csv(path, sep=',', header=None)
labels = samples.iloc[0:, 0].values
inputs = samples.iloc[0:, 1:].values

X_train, X_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.33, random_state=42)

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs

epochs = 50_000
input_dim = inputs.shape[1]
output_dim = 1  # Two possible outputs
learning_rate = 0.01

model = LogisticRegression(input_dim, output_dim)

criterion = torch.nn.BCELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

X_train, X_test = torch.Tensor(X_train), torch.Tensor(X_test)
y_train, y_test = torch.Tensor(y_train), torch.Tensor(y_test)

losses = []
losses_test = []
Iterations = []
iter = 0
for epoch in tqdm(range(int(epochs)), desc='Training Epochs'):
    x = X_train
    labels = y_train
    optimizer.zero_grad()  # Setting our stored gradients equal to zero
    outputs = model(X_train)
    loss = criterion(torch.squeeze(outputs), labels)  # [200,1] -squeeze-> [200]

    loss.backward()  # Computes the gradient of the given tensor w.r.t. graph leaves

    optimizer.step()  # Updates weights and biases with the optimizer (SGD)

    iter += 1
    if iter % 10000 == 0:
        # calculate Accuracy
        with torch.no_grad():
            # Calculating the loss and accuracy for the test dataset
            correct_test = 0
            total_test = 0
            outputs_test = torch.squeeze(model(X_test))
            loss_test = criterion(outputs_test, y_test)

            predicted_test = outputs_test.round().detach().numpy()
            total_test += y_test.size(0)
            correct_test += np.sum(predicted_test == y_test.detach().numpy())
            accuracy_test = 100 * correct_test / total_test
            losses_test.append(loss_test.item())

            # Calculating the loss and accuracy for the train dataset
            total = 0
            correct = 0
            total += y_train.size(0)
            correct += np.sum(torch.squeeze(outputs).round().detach().numpy() == y_train.detach().numpy())
            accuracy = 100 * correct / total
            losses.append(loss.item())
            Iterations.append(iter)

            print(f"Iteration: {iter}. \nTest - Loss: {loss_test.item()}. Accuracy: {accuracy_test}")
            print(f"Train -  Loss: {loss.item()}. Accuracy: {accuracy}\n")


# Print model's state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])

PATH = "/home/vincent/tmp/model-6656168128204039283"
torch.save(model.state_dict(), PATH)

def model_plot(model, X, y, title):
    parm = {}
    b = []
    for name, param in model.named_parameters():
        parm[name] = param.detach().numpy()

    w = parm['linear.weight'][0]
    b = parm['linear.bias'][0]
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='jet')
    u = np.linspace(X[:, 0].min(), X[:, 0].max(), 2)
    plt.plot(u, (0.5 - b - w[0] * u) / w[1])
    plt.xlim(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5)
    plt.ylim(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)
    plt.xlabel(r'$\boldsymbol{x_1}$',
               fontsize=16)  # Normally you can just add the argument fontweight='bold' but it does not work with latex
    plt.ylabel(r'$\boldsymbol{x_2}$', fontsize=16)
    plt.title(title)
    plt.show()


# Train Data
model_plot(model, X_train, y_train, 'Train Data')

# Test Dataset Results
model_plot(model, X_test, y_test, 'Test Data')
