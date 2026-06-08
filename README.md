# DL- Developing a Recurrent Neural Network Model for Stock Prediction

## AIM
To develop a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data.

## Problem Statement

The objective of this experiment is to develop a Recurrent Neural Network (RNN) model for predicting future stock prices using historical stock market data. The model learns temporal patterns from previous stock closing prices and predicts future closing prices.

### Dataset

The dataset consists of historical stock market records stored in:

- trainset.csv
- testset.csv

The 'Close' column is used as the target feature for prediction. The stock prices are normalized using MinMaxScaler and converted into sequences of 60 previous time steps for training the RNN model.

## DESIGN STEPS

STEP 1:
Import the required libraries such as NumPy, Pandas, Matplotlib, PyTorch, and Scikit-Learn.

STEP 2:
Load the stock market training and testing datasets and extract the closing price column.

STEP 3:
Normalize the stock prices using MinMaxScaler.

STEP 4:
Create input sequences using previous 60 closing prices to predict the next closing price.

STEP 5:
Convert the sequences into PyTorch tensors and create DataLoaders.

STEP 6:
Define the Recurrent Neural Network architecture using RNN and Fully Connected layers.

STEP 7:
Define the Mean Squared Error loss function and Adam optimizer.

STEP 8:
Train the RNN model using backpropagation through time.

STEP 9:
Predict stock prices on the testing dataset.

STEP 10:
Plot training loss and compare predicted stock prices with actual stock prices.

## PROGRAM

### Name: Arunsamy D

### Register Number: 212224240016

```python
# Define RNN Model
class RNNModel(nn.Module):

    def __init__(
            self,
            input_size=1,
            hidden_size=50,
            num_layers=2,
            output_size=1):

        super(RNNModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear( hidden_size, output_size )

    def forward(self, x):

        h0 = torch.zeros(
            self.num_layers,
            x.size(0),
            self.hidden_size
        ).to(x.device)

        out, _ = self.rnn(x, h0)

        out = out[:, -1, :]

        out = self.fc(out)

        return out

# Loss function

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# Train the Model

num_epochs = 50

train_losses = []

for epoch in range(num_epochs):

    model.train()

    running_loss = 0.0

    for inputs, targets in train_loader:

        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(
            outputs,
            targets
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    epoch_loss = (
        running_loss /
        len(train_loader)
    )

    train_losses.append(epoch_loss)

    print( f"Epoch [{epoch+1}/{num_epochs}] " f"Loss: {epoch_loss:.6f}" )

```

## OUTPUT

### Dataset Information
<img width="892" height="81" alt="image" src="https://github.com/user-attachments/assets/0e083d8a-cea9-411b-b778-f89184115e7b" />

### RNN Model Summary
<img width="1008" height="359" alt="image" src="https://github.com/user-attachments/assets/de29c9ac-79ea-473f-b228-3b38e4b6f0cd" />

### Training Loss Over Epochs Plot
<img width="645" height="992" alt="image" src="https://github.com/user-attachments/assets/f531fd16-f2d3-4719-bc24-4b2092575c23" />
<img width="654" height="558" alt="image" src="https://github.com/user-attachments/assets/4fc99038-1ef3-40cd-886b-8376bc58f6a1" />

### Predicted Stock Prices
<img width="859" height="547" alt="download" src="https://github.com/user-attachments/assets/31aa5ceb-1017-4a8a-901c-9e533a01d609" />

### Actual Stock Prices
<img width="640" height="61" alt="image" src="https://github.com/user-attachments/assets/0f6ac242-3a55-4579-84ee-7e1bd5df2c3e" />


### Predictions
<img width="910" height="1195" alt="image" src="https://github.com/user-attachments/assets/65b45ab7-d61b-4bd2-8be7-f2500c413b0e" />

## RESULT

Thus, a Recurrent Neural Network (RNN) model was successfully developed and trained using historical stock price data. The model learned temporal patterns from previous stock prices and was able to predict future stock prices with reasonable accuracy.
