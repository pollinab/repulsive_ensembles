from utils.mnist_config import configuration
import torch
from models.MLP import Net
from models.ensemble import Ensemble
from utils.evaluation import evaluate
from torchvision.datasets import MNIST
import torch.nn.functional as F
from torchvision import transforms
import os
import numpy as np
import warnings
from datetime import datetime
warnings.filterwarnings("ignore", category=DeprecationWarning) 

def run():
    """Run the script.
    """
    date = datetime.now().strftime('%H-%M-%S')
    os.makedirs('out', exist_ok=True)
    config = configuration()
    torch.manual_seed(config.random_seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('-------------------Prepare datasets------------------')

    batch_size = config.batch_size
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])
    train_set = MNIST(os.getcwd(), train=True, transform=transform, download=True)
    test_set = MNIST(os.getcwd(), train=False, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=len(test_set), shuffle=False)
    test_x, test_labels = next(iter(test_loader))


    ood_set = MNIST('data/notMNIST', train=False, transform=transform, download=False)
    ood_loader = torch.utils.data.DataLoader(ood_set, batch_size=len(ood_set), shuffle=False)
    ood_data, ood_labels = next(iter(ood_loader))

    n_particles = config.n_particles
    layer_sizes = [28*28, 100, 100, 100, 10]
    mnet = Net(layer_sizes, classification=True, act=F.relu, out_act=F.softmax, bias=True).to(device)
    l = []
    for _ in range(n_particles):
      l.append(torch.cat([p.flatten() for p in Net(layer_sizes, classification=True, act=F.relu, out_act=F.softmax, bias=True, no_weights=False).parameters()][len(mnet.param_shapes):]).detach())
    initial_particles = torch.stack(l).to(device)
    ensemble = Ensemble(device=device, net=mnet, particles=initial_particles)

    history = train(ensemble, train_loader, config)
    particles = ensemble.particles.cpu().detach().numpy()
    np.save('out/'+date+'particles', particles)

    print('-------------------Evaluation------------------')

    ent_ood, ent_test, metrics = evaluate(ensemble, test_x, test_labels, ood_data)
    np.save('out/'+date+'metrics', metrics)

    
if __name__ == '__main__':
    run()

