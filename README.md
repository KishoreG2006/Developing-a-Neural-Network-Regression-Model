# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Explain the problem statement

## Neural Network Model

<img width="1096" height="733" alt="image" src="https://github.com/user-attachments/assets/bc9c13ef-1c58-4bda-a37a-b89a6f135a13" />


## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name:KISHORE G

### Register Number: 212223040099

```python
# Name:KISHORE G
# Register Number:212223040099
class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1,8)
        self.fc2 = nn.Linear(8,10)
        self.fc3 = nn.Linear(10,1)
        self.relu = nn.ReLU()
        self.history = {'loss':[]}

  def forward(self,x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x



lig = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.RMSprop (lig. parameters(),lr=0.001)



# Name:KISHORE G
# Register Number:212223040099
def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
  for epoch in range (epochs):
    optimizer. zero_grad()
    loss = criterion(ai_brain(X_train),y_train)
    loss. backward()
    optimizer.step()
    lig.history['loss'].append(loss.item())
    if epoch % 200 == 0:
      print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')

```

### Dataset Information

<img width="176" height="237" alt="image" src="https://github.com/user-attachments/assets/ebb9edd5-0459-4321-ba0b-b61909f03af8" />



### OUTPUT

<img width="346" height="234" alt="image" src="https://github.com/user-attachments/assets/92f90490-0fef-4f4b-ad6e-e361d764b207" />


### Training Loss Vs Iteration Plot

<img width="713" height="571" alt="image" src="https://github.com/user-attachments/assets/0a134d7c-6fbe-40d7-b888-848897f33c48" />

### New Sample Data Prediction

<img width="1221" height="135" alt="image" src="https://github.com/user-attachments/assets/52036f55-2437-40f7-bc24-606a24f52b0e" />

## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
