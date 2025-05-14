import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def get_mnist_dataset(batch_size=64):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                              pin_memory=True if torch.cuda.is_available() else False)
    return train_loader


class DigitReconstructor(nn.Module):
    def __init__(self):
        super(DigitReconstructor, self).__init__()

        self.encoder = nn.Sequential(

            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),  # 14x14

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),  # 7x7

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),  # 3x3
        )

        self.flatten = nn.Flatten()

        self.bottleneck = nn.Sequential(
            nn.Linear(256 * 3 * 3, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),

            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),

            nn.Linear(64, 10)
        )

        self.decoder_fc = nn.Sequential(
            nn.Linear(10, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),

            nn.Linear(64, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.1),

            nn.Linear(1024, 784),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        features = self.bottleneck(x)
        return features

    def decode(self, features):
        x = self.decoder_fc(features)
        return x.view(-1, 1, 28, 28)

    def forward(self, x):
        features = self.encode(x)
        reconstructed = self.decode(features)
        return features, reconstructed


def train_autoencoder(max_epochs=10, learning_rate=0.001, batch_size=128):
    train_loader = get_mnist_dataset(batch_size=batch_size)

    model = DigitReconstructor().to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=2, eta_min=1e-5)

    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    all_losses = []
    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)

            optimizer.zero_grad()

            if scaler is not None:
                with torch.cuda.amp.autocast():
                    features, reconstructed = model(data)

                    loss = criterion(reconstructed, data) + 1e-4 * torch.norm(features, 1)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                features, reconstructed = model(data)

                loss = criterion(reconstructed, data) + 1e-4 * torch.norm(features, 1)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            running_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch + 1}/{max_epochs}, Batch {batch_idx}/{len(train_loader)}, "
                      f"Loss: {running_loss / (batch_idx + 1):.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

        epoch_loss = running_loss / len(train_loader)
        all_losses.append(epoch_loss)

        print(f"Epoch {epoch + 1} complete, Loss: {epoch_loss:.6f}")

        scheduler.step()

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

    torch.save(model.state_dict(), 'digit_reconstructor.pth')
    print("Model saved to digit_reconstructor.pth")

    return model


def visualize_reconstructions(model, data_loader, epoch):
    model.eval()

    data_iter = iter(data_loader)
    images, _ = next(data_iter)
    images = images[:10].to(device)

    with torch.no_grad():
        _, reconstructed = model(images)

    images = images.cpu()
    reconstructed = reconstructed.cpu()

    fig, axes = plt.subplots(2, 10, figsize=(20, 4))
    fig.suptitle(f"Original vs Reconstructed (Epoch {epoch + 1})", fontsize=16)

    for i in range(10):
        original = images[i].squeeze(0).numpy()
        original = original * 0.5 + 0.5
        axes[0, i].imshow(original, cmap='gray')
        axes[0, i].set_title(f"Original {i}")
        axes[0, i].axis('off')

        recon = reconstructed[i].squeeze(0).numpy()
        recon = recon * 0.5 + 0.5
        axes[1, i].imshow(recon, cmap='gray')
        axes[1, i].set_title(f"Reconstructed {i}")
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig(f"reconstructions_epoch_{epoch + 1}.png")
    plt.close()
    print(f"Saved reconstructions to reconstructions_epoch_{epoch + 1}.png")


def visualize_digit_dimensions(model, epoch):
    model.eval()

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle(f"Digit Patterns (Epoch {epoch + 1})", fontsize=16)

    amplification_values = [1.0, 3.0, 5.0, 8.0]
    best_patterns = []
    best_amp = 5.0

    with torch.no_grad():
        for amp in amplification_values:
            patterns = []
            for i in range(10):
                one_hot = torch.zeros(1, 10).to(device)
                one_hot[0, i] = amp
                reconstructed = model.decode(one_hot)
                digit_pattern = reconstructed.view(28, 28).cpu().numpy()
                patterns.append(digit_pattern)

            avg_contrast = np.mean([np.max(p) - np.min(p) for p in patterns])
            if avg_contrast > np.mean([np.max(p) - np.min(p) for p in best_patterns]) or not best_patterns:
                best_patterns = patterns
                best_amp = amp

    print(f"Using amplification value: {best_amp}")

    for i, ax in enumerate(axes.flat):
        if i < len(best_patterns):
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

    import os

    os.makedirs('images', exist_ok=True)

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    np.random.seed(42)

    trained_model = train_autoencoder(max_epochs=10, learning_rate=0.001, batch_size=128)

    print("Training complete!")