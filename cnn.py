import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_mnist_dataset(batch_size=64):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader


class CNNWithFC(nn.Module):
    def __init__(self):
        super(CNNWithFC, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.flatten(start_dim=1)
        x = self.fc1(x)
        return x


def train_and_visualize_weights(max_epochs=5):
    train_loader = get_mnist_dataset(batch_size=64)

    model = CNNWithFC()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    visualize_fc_weights(model, "Random Weights (Before Training)", "images/weights_random")

    for epoch in range(max_epochs):
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 100 == 0:
                print(
                    f"Epoch {epoch + 1}/{max_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {running_loss / (batch_idx + 1):.4f}")

        print(f"Epoch {epoch + 1} complete, Loss: {running_loss / len(train_loader):.4f}")

        visualize_fc_weights(model, f"Weights After Epoch {epoch + 1}", f"images/weights_epoch_{epoch + 1}")

        visualize_activations_for_digits(model, train_loader, epoch)


def visualize_fc_weights(model, title, filename):
    weights = model.fc1.weight.data.cpu().numpy()

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle(title, fontsize=16)

    for i, ax in enumerate(axes.flat):
        weight_img = weights[i].reshape(64, 7, 7)

        avg_activation = np.mean(weight_img, axis=0)

        avg_activation = (avg_activation - avg_activation.min()) / (avg_activation.max() - avg_activation.min() + 1e-9)

        im = ax.imshow(avg_activation, cmap='viridis')
        ax.set_title(f"Digit {i}")
        ax.axis('off')

    plt.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7)
    plt.tight_layout()
    plt.savefig(f"{filename}.png")
    plt.close()
    print(f"Saved weight visualization to {filename}.png")


def visualize_activations_for_digits(model, data_loader, epoch):
    digit_examples = {}
    for data, targets in data_loader:
        for img, target in zip(data, targets):
            target_val = target.item()
            if target_val not in digit_examples:
                digit_examples[target_val] = img

            if len(digit_examples) == 10:
                break
        if len(digit_examples) == 10:
            break

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle(f"Model Activation Maps After Epoch {epoch + 1}", fontsize=16)

    for i, ax in enumerate(axes.flat):
        if i in digit_examples:
            img = digit_examples[i].unsqueeze(0)

            with torch.no_grad():
                act1 = torch.relu(model.conv1(img))

            act1_mean = act1.squeeze(0).mean(0).cpu().numpy()

            act1_norm = (act1_mean - act1_mean.min()) / (act1_mean.max() - act1_mean.min() + 1e-9)

            im = ax.imshow(act1_norm, cmap='viridis')
            ax.set_title(f"Digit {i}")
            ax.axis('off')

    plt.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7)
    plt.tight_layout()
    plt.savefig(f"images/activation_maps_epoch_{epoch + 1}.png")
    plt.close()
    print(f"Saved activation maps to activation_maps_epoch_{epoch + 1}.png")


def visualize_digit_dimensions(model, epoch):
    """Visualize what the network associates with each digit dimension"""
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle(f"Reconstructed Digit Patterns (Epoch {epoch + 1})", fontsize=16)

    for i, ax in enumerate(axes.flat):
        one_hot = torch.zeros(1, 10)
        one_hot[0, i] = 3.0

        with torch.no_grad():
            reconstructed = torch.sigmoid(model.fc_decode(one_hot))
            digit_pattern = reconstructed.view(28, 28).cpu().numpy()

        im = ax.imshow(digit_pattern, cmap='gray')
        ax.set_title(f"Digit {i}")
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(f"images/digit_patterns_epoch_{epoch + 1}.png")
    plt.close()
    print(f"Saved digit patterns to digit_patterns_epoch_{epoch + 1}.png")


if __name__ == "__main__":
    print("Starting training and weight visualization...")
    train_and_visualize_weights(max_epochs=2)
