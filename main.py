import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from VAE import VAE
from tqdm import tqdm
import matplotlib.pyplot as plt



device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
# device = 'cpu'
print(device)

# def calc_loss(x_recon, x, mu, sigma):
#     recon_loss = F.binary_cross_entropy(x_recon, x)
#     kl_div = -0.5 * torch.mean(1 + sigma - mu.pow(2) - sigma.exp())
#     return recon_loss+kl_div, recon_loss, kl_div


def train(model, data):
    model.train()

    learning_rate = 1e-3
    num_epochs = 10
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterian = nn.MSELoss()
    for epoch in range(num_epochs):
        loop = tqdm(data)
        for i, (x, y) in enumerate(loop):
            x = x.to(device)
            x_reconstructed, mu, sigma = model(x)

            recon_loss = criterian(x_reconstructed, x)
            kl_div = -0.5 * torch.mean(1 + sigma - mu.pow(2) - sigma.exp())
            loss = recon_loss + kl_div

            # loss, recon_loss, kl_div = calc_loss(x_reconstructed, x, mu, sigma)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop.set_postfix(loss = loss.item())
        torch.save(model.state_dict(), f'checkpoint/VAE_{num_epochs}.pt')



def main():
    print("In side main")
    train_data = dataset.ImageFolder('data/train', transform=transforms.Compose([
                                                        transforms.Resize([128, 128]),
                                                        transforms.ToTensor()]))

    train_loader = DataLoader(dataset=train_data, batch_size=32, shuffle=True)

    in_channels = 3
    latent_dim = 20

    model = VAE(in_channels=in_channels, latent_dim=latent_dim)
    model = model.to(device)
    train(model, data=train_loader)

def test():
    print("test")
    model = VAE(3, 20)
    model = model.to(device)
    model_state = torch.load('checkpoint/VAE_10.pt')
    model.load_state_dict(model_state)
    model.eval()

    val_data = dataset.ImageFolder('data/val', transform=transforms.Compose([
                                                        transforms.Resize([128, 128]),
                                                        transforms.ToTensor()]))
    
    val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=True)
    loop = tqdm(val_loader)
    for i, (x, y) in enumerate(loop):
        x = x.to(device)
        x_reconstructed, mu, sigma = model(x)
        break

    
    print(x_reconstructed.shape)
    x_1 = x_reconstructed[0].cpu().detach()
    x_1_plot = x_1.permute(1,2,0).numpy()
    plt.imshow(x_1_plot)
    plt.savefig('op1.jpg')


if __name__ == "__main__":
    test()



