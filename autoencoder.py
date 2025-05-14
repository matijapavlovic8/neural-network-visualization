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


class DigitReconstructor(nn.Module):
    def __init__(self):
        super(DigitReconstructor, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.fc_encode1 = nn.Linear(256 * 3 * 3, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.fc_encode2 = nn.Linear(512, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)
        self.fc_encode3 = nn.Linear(128, 10)

        self.fc_decode1 = nn.Linear(10, 128)
        self.bn_dec1 = nn.BatchNorm1d(128)
        self.fc_decode2 = nn.Linear(128, 512)
        self.bn_dec2 = nn.BatchNorm1d(512)
        self.fc_decode3 = nn.Linear(512, 1024)
        self.bn_dec3 = nn.BatchNorm1d(1024)
        self.fc_decode4 = nn.Linear(1024, 784)

        self.dropout = nn.Dropout(0.2)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.activation(self.bn1(self.conv1(x)))
        x = torch.max_pool2d(x, 2)
        x = self.activation(self.bn2(self.conv2(x)))
        x = torch.max_pool2d(x, 2)
        x = self.activation(self.bn3(self.conv3(x)))
        x = torch.max_pool2d(x, 2)

        x = x.flatten(start_dim=1)

        e1 = self.activation(self.bn_fc1(self.fc_encode1(x)))
        e1 = self.dropout(e1)
        e2 = self.activation(self.bn_fc2(self.fc_encode2(e1)))
        e2 = self.dropout(e2)
        features = self.fc_encode3(e2)

        d1 = self.activation(self.bn_dec1(self.fc_decode1(features)))
        d1 = self.dropout(d1)
        d2 = self.activation(self.bn_dec2(self.fc_decode2(d1)))
        d2 = self.dropout(d2)
        d3 = self.activation(self.bn_dec3(self.fc_decode3(d2)))
        d3 = self.dropout(d3)
        reconstructed = torch.sigmoid(self.fc_decode4(d3))

        return features, reconstructed.view(-1, 1, 28, 28)


def train_autoencoder(max_epochs=10, learning_rate=0.001, batch_size=128):
    train_loader = get_mnist_dataset(batch_size=batch_size)

    model = DigitReconstructor()
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1, factor=0.7)

    all_losses = []
    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (data, _) in enumerate(train_loader):
            optimizer.zero_grad()

            _, reconstructed = model(data)

            loss = criterion(reconstructed, data)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch + 1}/{max_epochs}, Batch {batch_idx}/{len(train_loader)}, "
                      f"Loss: {running_loss / (batch_idx + 1):.6f}")

        epoch_loss = running_loss / len(train_loader)
        all_losses.append(epoch_loss)

        print(f"Epoch {epoch + 1} complete, Loss: {epoch_loss:.6f}")

        scheduler.step(epoch_loss)

        if (epoch + 1) % 2 == 0 or epoch == 0 or epoch == max_epochs - 1:
            visualize_reconstructions(model, train_loader, epoch)
            visualize_digit_dimensions(model, epoch)

    plt.figure(figsize=(10, 6))
    plt.plot(all_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True)
    plt.savefig('images/autoencoder_training_loss.png')
    plt.close()

    return model


def visualize_reconstructions(model, data_loader, epoch):
    model.eval()

    images, _ = next(iter(data_loader))
    images = images[:10]

    with torch.no_grad():
        _, reconstructed = model(images)

    fig, axes = plt.subplots(2, 10, figsize=(20, 4))
    fig.suptitle(f"Original vs Reconstructed (Epoch {epoch + 1})", fontsize=16)

    for i in range(10):
        original = images[i].squeeze(0).cpu().numpy()
        original = original * 0.5 + 0.5
        axes[0, i].imshow(original, cmap='gray')
        axes[0, i].set_title(f"Original {i}")
        axes[0, i].axis('off')

        recon = reconstructed[i].squeeze(0).cpu().numpy()
        recon = recon * 0.5 + 0.5
        axes[1, i].imshow(recon, cmap='gray')
        axes[1, i].set_title(f"Reconstructed {i}")
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig(f"images/reconstructions_epoch_{epoch + 1}.png")
    plt.close()
    print(f"Saved reconstructions to reconstructions_epoch_{epoch + 1}.png")


def visualize_digit_dimensions(model, epoch):
    model.eval()

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle(f"Digit Patterns (Epoch {epoch + 1})", fontsize=16)

    amplification_values = [1.0, 2.0, 3.0, 5.0]
    best_patterns = []
    best_amp = 3.0

    with torch.no_grad():
        for amp in amplification_values:
            patterns = []
            for i in range(10):
                one_hot = torch.zeros(1, 10)
                one_hot[0, i] = amp

                x = model.activation(model.bn_dec1(model.fc_decode1(one_hot)))
                x = model.activation(model.bn_dec2(model.fc_decode2(x)))
                x = model.activation(model.bn_dec3(model.fc_decode3(x)))
                reconstructed = torch.sigmoid(model.fc_decode4(x))

                digit_pattern = reconstructed.view(28, 28).cpu().numpy()
                patterns.append(digit_pattern)
            avg_contrast = np.mean([np.max(p) - np.min(p) for p in patterns])
            if avg_contrast > np.mean([np.max(p) - np.min(p) for p in best_patterns]) or not best_patterns:
                best_patterns = patterns
                best_amp = amp

    print(f"Using amplification value: {best_amp}")

    for i, ax in enumerate(axes.flat):
        digit_pattern = best_patterns[i]

        im = ax.imshow(digit_pattern, cmap='gray')
        ax.set_title(f"Digit {i}")
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(f"images/digit_patterns_epoch_{epoch + 1}.png")
    plt.close()
    print(f"Saved digit patterns to digit_patterns_epoch_{epoch + 1}.png")


if __name__ == "__main__":
    print("Training the enhanced autoencoder for digit visualization...")
    trained_model = train_autoencoder(max_epochs=10, learning_rate=0.001, batch_size=128)

    print("Training complete!")