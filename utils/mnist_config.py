from datetime import datetime
import argparse

def configuration(args = None):
    parser = argparse.ArgumentParser(description='Repulsive deep ensembles.')

    tgroup = parser.add_argument_group('Training options')
    tgroup.add_argument('--n_epochs', type=int, metavar='N', default=15,
                        help='Number of training epochs. '+
                             'Default: %(default)s.')
    tgroup.add_argument('--batch_size', type=int, metavar='N', default=128,
                        help='Training batch size. Default: %(default)s.')
    tgroup.add_argument('--lr', type=float, default=1e-2,
                        help='Learning rate of optimizer. Default: ' +
                             '%(default)s.')
    tgroup.add_argument('--method', type=str, default='kde',
                        help='Method for optimization, options: kde, sge, ssge. Default: ' +
                             '%(default)s.')
    tgroup.add_argument('--kernel', type=str, default='rbf',
                        help='Type of kernel, options: rbf, laplace, linear. Default: ' +
                             '%(default)s.')
    tgroup.add_argument('--functional', type=bool, default=False,
                        help='Whether to use otimization in functional space. Default: ' +
                             '%(default)s.')
    tgroup.add_argument('--adversarial', type=bool, default=False,
                        help='Whether to use adversarial examples. Default: ' +
                             '%(default)s.')
    tgroup.add_argument('--n_particles', type=int, default=5,
                        help='Number of particles used for the approximation of the gradient flow')

    tgroup.add_argument('--anneal', type=float, default=0.01, help='Annealing factor '+'Default: %(default)s.')
    tgroup.add_argument('--eps', type=float, default=0.001, help='Parameter for adversarial update '+'Default: %(default)s.')
    tgroup.add_argument('--gamma', type=float, default=0.8, help='Scheduler parameter '+'Default: %(default)s.')
    tgroup.add_argument('--kernel_param', type=float, default=100, help='Kernel parameter '+'Default: %(default)s.')

    
    
    if args is None:
        return parser.parse_args()
    else:
        return parser.parse_args(args = args)
