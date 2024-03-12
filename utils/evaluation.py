import uncertainty_metrics as um
import seaborn as sns
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate(ensemble, test_x, test_labels, ood_data, num_classes=10):
  with torch.no_grad():
    test_x = test_x.reshape(len(test_x), -1).to(device)
    test_labels = test_labels.to(device)

    test_pred = ensemble.forward(test_x)
    Y_t = torch.mean(test_pred[0], 0)

    log_scale = torch.log2(torch.tensor(num_classes ,dtype=torch.float))
    entropies_test = -torch.sum(torch.log2(Y_t + 1e-20)/log_scale * Y_t, 1).detach().cpu()

    test_accuracy = (torch.argmax(Y_t, 1) == test_labels).sum().item() / Y_t.shape[0] * 100
    nll = F.nll_loss(torch.log(Y_t), test_labels).item()
    test_labels_np = test_labels.cpu().detach().numpy().astype(np.int8)
    test_ece = um.numpy.ece(test_labels_np, Y_t.cpu().detach().numpy(), num_bins=30)

    ood_data = ood_data.reshape(len(ood_data), -1).to(device)
    ood_pred = ensemble.forward(ood_data.to(device))[0]

    average_prob_ood = ood_pred.mean(0)
    entropies_ood = -torch.sum(torch.log2(average_prob_ood + 1e-20)/log_scale * average_prob_ood, 1).detach().cpu()
    entropies_ratio = (entropies_ood.mean() / entropies_test.mean()).item()

    sns.kdeplot(entropies_ood, label='ood')
    sns.kdeplot(entropies_test, label='test')
    plt.legend()
    plt.show()

    print('Test accuracy:', test_accuracy, ', nll loss:', nll,
          ', test ECE:', test_ece, ', entropies ratio:', entropies_ratio)

  return entropies_ood, entropies_test, [test_accuracy, nll, test_ece, entropies_ratio]

