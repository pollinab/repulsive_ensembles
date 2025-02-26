# Repulsive Deep Ensembles are Bayesian
Project based on the paper [Repulsive deep ensembles are Bayesian](https://proceedings.neurips.cc/paper/2021/hash/1c63926ebcabda26b5cdb31b5cc91efb-Abstract.html)

This is a forked version of the authors' repository. The functional that we added for our experiments: 
- training pipeline -- training/train_classifier.py
- laplace kernel -- utils/kernel.py
- evaluation on test and ood data  -- utils/evaluation.py
- visualization -- utils/visualization.py
- script to run experiments -- experiments/exp_mnist.py + utils/mnist_config.py
  
## MNIST experiment

Training on MNIST and evaluation on NotMNIST. Example run:

```console
$ gzip -d data/notMNIST/MNIST/raw/t10k-labels-idx1-ubyte.gz
$ gzip -d data/notMNIST/MNIST/raw/t10k-images-idx3-ubyte.gz
$ python setup.py install
$ python experiments/exp_mnist.py --n-epochs 20 --n_particles 20 --method kde --kernel laplace --functional False --adversarial True
```
For more training options, see 'utils/mnist_config.py'

To train in GoogleColab see 'notebooks/training.ipynb'.

## Visualization

Notebooks 'notebooks/training.ipynb' and 'notebooks/visualization.ipynb' contain evaluation and visualization examples. We experiment with test (MNIST) and ood data (NotMNIST, RotatedMNIST) 

## Citation
```
@article{d2021repulsive,
  title={Repulsive Deep Ensembles are Bayesian},
  author={D'Angelo, Francesco and Fortuin, Vincent},
  journal={arXiv preprint arXiv:2106.11642},
  year={2021}
}
```
