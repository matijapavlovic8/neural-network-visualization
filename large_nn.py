import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

def get_dataset():
    X, y = make_moons(n_samples=500, noise=0.35, random_state=0)
    return train_test_split(X, y, test_size=0.3, random_state=42)

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

def train_and_animate(hidden_size, title, filename, max_epochs=300, interval=100):
    X_train, X_test, y_train, y_test = get_dataset()
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.long)

    model = SimpleNN(hidden_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    loss_history = []

    x_min, x_max = X_train[:, 0].min() - .5, X_train[:, 0].max() + .5
    y_min, y_max = X_train[:, 1].min() - .5, X_train[:, 1].max() + .5
    h = 0.02
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    input_grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    decision_ax, loss_ax = axs

    frames = max_epochs // 10

    def update(epoch):
        for _ in range(10):
            optimizer.zero_grad()
            output = model(X_tensor)
            loss = criterion(output, y_tensor)
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())

        decision_ax.clear()
        with torch.no_grad():
            Z = torch.argmax(model(input_grid), axis=1).reshape(xx.shape)
        decision_ax.contourf(xx, yy, Z.numpy(), cmap=plt.cm.Spectral, alpha=0.8)
        decision_ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Set1, edgecolor='k')
        decision_ax.set_title(f"{title} — Epoch {epoch * 10}")

        loss_ax.clear()
        loss_ax.plot(loss_history, label="Training Loss", color='red')
        loss_ax.set_title("Loss Curve")
        loss_ax.set_xlabel("Iterations")
        loss_ax.set_ylabel("Loss")
        loss_ax.legend()

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=interval, repeat=False)
    globals()[f"_ani_{filename}"] = ani
    ani.save(f"{filename}.gif", writer='pillow', fps=5)
    print(f"Saved animation to {filename}.gif")

if __name__ == "__main__":
    train_and_animate(hidden_size=5, title="Underfitting (Small NN)", filename="images/underfit_moons")
    train_and_animate(hidden_size=50, title="Good Fit (Larger NN)", filename="images/fit_moons")
