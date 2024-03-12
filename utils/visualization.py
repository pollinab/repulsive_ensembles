from torchvision.utils import make_grid
import torch
import numpy as np
import matplotlib.pyplot as plt

def show_chosen_samples(score, data, probs=None, k=10, title=''):
  """
  Display samples with low and high scores
  """

  data = data.reshape(len(data), 1, 28, 28)
  values, ind = torch.sort(score)
  high_confidence_samples = data[ind[:k]]
  if probs is not None:
    print("predicted classes for 1 row")
    print(torch.argmax(probs[ind[:k]], -1)) #[3, 3, 3, 3, 3, 0, 0, 7, 3, 3]

  #middle_confidence_samples = data[ind[len(ind)//2:len(ind)//2 + k]]
  middle_confidence_samples = data[ind[1000:1000+k]]
  low_confidence_samples = data[ind[-k-800:-800]]
  least_confidence_samples = data[ind[-k:]]

  image_grid = make_grid(torch.vstack([high_confidence_samples,
                                       middle_confidence_samples,
                                       low_confidence_samples,
                                       least_confidence_samples]), nrow=k)
  plt.imshow(image_grid.permute(1, 2, 0))
  plt.title(title)
  plt.show()


def plot_ordered_data(ensemble, data, num_classes=10, k=10):
  """
  High and low confidence samples according to the ensemble
  """

  with torch.no_grad():
    data = data.reshape(len(data), -1).to(device)

    test_pred = ensemble.forward(data)
    Y_t = torch.mean(test_pred[0], 0)
    std_prob_test = test_pred[0].std(0).mean(1)

    log_scale = torch.log2(torch.tensor(num_classes ,dtype=torch.float))
    entropies_test = -torch.sum(torch.log2(Y_t + 1e-20)/log_scale * Y_t, 1).detach().cpu()

    data = data.reshape(len(data), 1, 28, 28)
    show_chosen_samples(score=entropies_test, data=data, k=10, probs=Y_t, title='entropy')
    show_chosen_samples(score=-torch.max(Y_t, -1)[0], data=data, k=10, probs=Y_t, title='max prob')
    show_chosen_samples(score=std_prob_test, data=data, k=10, probs=Y_t, title='std')

def plot_calibration(ensemble, data, labels, num_classes=10):
  """
  Plot accuracy vs confidence threshold
  """

  with torch.no_grad():
    data = data.reshape(len(data), -1).to(device)

    test_pred = ensemble.forward(data)
    Y_t = torch.mean(test_pred[0], 0)

    log_scale = torch.log2(torch.tensor(num_classes ,dtype=torch.float))
    entropies_test = -torch.sum(torch.log2(Y_t + 1e-20)/log_scale * Y_t, 1).detach().cpu()

    rule = torch.max(Y_t, -1)[0]
    acc = []
    #correct_sorted = (torch.argmax(Y_t, 1) == labels)[indices.squeeze()]
    for threshold in np.linspace(0.1, 1., 10):
        acc.append(torch.mean((torch.argmax(Y_t, 1) == labels)[rule >= threshold]*1.))

    plt.plot(np.linspace(0.1, 1., 10), acc, markersize=4, c='b', marker="o")
    plt.title(r'Accuracy on samples where $p(y|x) \geq \tau$')
    plt.xlabel(r'Confidence threshold $\tau$')
    plt.ylabel('accuracy')
    plt.show()

