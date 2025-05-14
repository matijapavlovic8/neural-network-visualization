import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.animation as animation
import numpy as np

X, y = make_moons(n_samples=300, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)

class SimpleNN(nn.Module):
    def __init__(self, hidden_size):
        super(SimpleNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2)
        )

    def forward(self, x):
        return self.net(x)

def animate_training(hidden_size=10, max_epochs=300, interval=100):
    model = SimpleNN(hidden_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    fig, ax = plt.subplots()
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.02
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    input_grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

    frames = max_epochs // 10

    def update(epoch):
        for _ in range(10):
            optimizer.zero_grad()
            output = model(X_train_tensor)
            loss = criterion(output, y_train_tensor)
            loss.backward()
            optimizer.step()

        ax.clear()
        with torch.no_grad():
            Z = torch.argmax(model(input_grid), axis=1).reshape(xx.shape)
        ax.contourf(xx, yy, Z.numpy(), cmap=plt.cm.Spectral, alpha=0.8)
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolor='k')
        ax.set_title(f"Epoch {epoch * 10}")

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=interval, repeat=False)

    globals()["_ani"] = ani
    ani.save("images/training_animation.gif", writer='pillow', fps=5)
    print("GIF saved as training_animation.gif")

animate_training(hidden_size=20, max_epochs=300, interval=100)
