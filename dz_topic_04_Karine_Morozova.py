# %%
import pandas as pd

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch import nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import  r2_score

import warnings

# %%
# filter warnings
warnings.filterwarnings('ignore')

# %%

df = pd.read_csv('./ConcreteStrengthData.csv')
    
# %%
df.columns

# %%
components = ['CementComponent ',
              'BlastFurnaceSlag',
              'FlyAshComponent',
              'WaterComponent',
              'SuperplasticizerComponent',
              'CoarseAggregateComponent',
              'FineAggregateComponent']

df['Components'] = df[components].gt(0).sum(axis=1)

df[components + ['Components']].head(10)

# %%
X = df.drop('Strength', axis=1)
y= df['Strength']

# %%

X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                    random_state=42,
                                                    test_size=0.3)

# %%
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# %%

class LinearModel(nn.Module):
    def __init__(self, in_dim, out_dim=1):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            
            nn.Linear(64, out_dim),
        )
        
    def forward(self, x):
        output = self.features(x)
        return output

# %%

model = LinearModel(in_dim=X_train.shape[1], out_dim=1)
  
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
    
# train_losses = []
# train_rmses = []
# test_losses = []
# test_rmses = []

# %%

# class ConcreteDataset(Dataset):
#     def __init__(self, X, y, scale=True):        
#         self.X = X.values # from Pandas DataFrame to NumPy array
#         self.y = y
        
#         if scale:
#             sc = StandardScaler()
#             self.X = sc.fit_transform(self.X)

#     def __len__(self):
#         #return size of a dataset
#         return len(self.y)

#     def __getitem__(self, idx):
#         #supports indexing using dataset[i] to get the i-th row in a dataset
        
#         X = torch.tensor(self.X[idx], dtype=torch.float32)
#         y = torch.tensor(self.y[idx], dtype=torch.float32)        
        
#         return X, y
batch_size = 32
num_epochs = 100

# 5. Model Training
# 5.1 Create custom Dataset and DataLoader for training and test sets
class ConcreteDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
     # %%
train_dataset = ConcreteDataset(X_train, y_train)
test_dataset = ConcreteDataset(X_test, y_test)

# %%

train_dataloader = DataLoader(train_dataset,
                              batch_size=batch_size, 
                              shuffle=True
                             )

test_dataloader = DataLoader(test_dataset,
                              batch_size=batch_size
                             )
    
# %%

train_losses, test_losses = [], []
train_rmses, test_rmses = [], []
train_maes, test_maes = [], []
train_r2s, test_r2s = [], []

# %%

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    y_pred_train = []
    
    for inputs, targets in train_dataloader:
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets.unsqueeze(1))  # Match dimensions
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        y_pred_train.extend(outputs.cpu().detach().numpy())
    
    # Calculate RMSE for training set
    # train_rmse = MSE(y_train, y_pred_train, squared=False)
    # train_rmses.append(train_rmse)
    train_rmse = MSE(y_train, y_pred_train, squared=False)
    train_mae = MAE(y_train, y_pred_train)
    train_r2 = r2_score(y_train, y_pred_train)
    train_rmses.append(train_rmse)
    train_maes.append(train_mae)
    train_r2s.append(train_r2)
    
    train_losses.append(train_loss / len(train_dataloader))

    # Evaluation step
    model.eval()
    test_loss = 0
    y_pred_test = []

    with torch.no_grad():
        for inputs, targets in test_dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, targets.unsqueeze(1))
            
            test_loss += loss.item()
            y_pred_test.extend(outputs.cpu().numpy())

    # Calculate RMSE for test set
    # test_rmse = MSE(y_test, y_pred_test, squared=False)
    # test_rmses.append(test_rmse)
    test_rmse = MSE(y_test, y_pred_test, squared=False)
    test_mae = MAE(y_test, y_pred_test)
    test_r2 = r2_score(y_test, y_pred_test)
    test_rmses.append(test_rmse)
    test_maes.append(test_mae)
    test_r2s.append(test_r2)
    
    test_losses.append(test_loss / len(test_dataloader))
    
    # Print progress every 10 epochs
    if (epoch + 1) % 5 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Train RMSE: {train_rmse:.4f}, Test Loss: {test_losses[-1]:.4f}, Test RMSE: {test_rmse:.4f}')

# %%
# plt.figure(figsize=(4, 3))
# plt.plot(train_losses, label='Train')
# plt.plot(test_losses, label='Validation')
# plt.legend(loc='best')
# plt.xlabel('Epochs')
# plt.ylabel('MAE')
# plt.title('Training vs Validation Loss')
# plt.show()


# 6. Model Evaluation
# 6.1 Plot loss curves
plt.figure(figsize=(6, 4))
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Validation Loss')
plt.legend(loc='best')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.title('Training vs Validation Loss')
plt.show()

# %%    
# 6.3 Plot MAE curves
plt.figure(figsize=(6, 4))
plt.plot(train_maes, label='Train MAE')
plt.plot(test_maes, label='Validation MAE')
plt.legend(loc='best')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.title('Training vs Validation MAE')
plt.show()   

# %%
# 6.4 Plot R² curves
plt.figure(figsize=(6, 4))
plt.plot(train_r2s, label='Train R²')
plt.plot(test_r2s, label='Validation R²')
plt.legend(loc='best')
plt.xlabel('Epochs')
plt.ylabel('R²')
plt.title('Training vs Validation R²')
plt.show()
    
# %%
print("Final Metrics:")
print(f"Train MSE: {train_losses[-1]:.4f}")
print(f"Train MAE: {train_maes[-1]:.4f}")
print(f"Train R²: {train_r2s[-1]:.4f}\n")

print(f"Train MSE: {test_losses[-1]:.4f}")
print(f"Test MAE: {test_maes[-1]:.4f}")
print(f"Test R²: {test_r2s[-1]:.4f}")   
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    