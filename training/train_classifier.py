import torch
from utils.SSGE_squeeze import SpectralSteinEstimator
from utils.kernel import RBF, Laplace, Linear
import torch.nn.functional as F

ssge_k = RBF()
ssge = SpectralSteinEstimator(0.01, None, ssge_k, device=device)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def repulsive_term(param, kernel, method, num_models):
  if num_models == 1 or method == 'none':
      return torch.zeros(1, device=device)

  if method == 'kde':
      K = kernel.forward(param, param.detach())
      grad_K = torch.autograd.grad(K.sum(), param)[0]
      grad_K = grad_K.view(num_models,-1)

      return grad_K / K.sum(1,keepdim=True)

  if method == 'ssge':
      return ssge.compute_score_gradients(param, param)

  if method == 'sge':
      eta = 0.01
      K = kernel.forward(param, param.detach())
      grad_K = torch.autograd.grad(K.sum(), param)[0]
      grad_K = grad_K.view(num_models,-1)

      K_ = K + eta * torch.eye(K.shape[0]).to(device)
      return torch.linalg.solve(K_,grad_K)



def train(ensemble, train_dataloader, config):

  if config.kernel == 'rbf':
    kernel = RBF()
  elif config.kernel == 'laplace':
    kernel = Laplace(s=config.kernel_param)
  else:
    kernel = Linear()
  
  num_train = len(train_loader.dataset)
  W = ensemble.particles
  optimizer = torch.optim.Adam([W], config.lr, weight_decay=0.,
                               betas=[0.9, 0.999])

  scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.gamma)

  priors = []
  for i in range(len(W)):
    priors.append(torch.distributions.normal.Normal(torch.zeros(ensemble.net.num_params).to(device),
                                                     torch.ones(ensemble.net.num_params).to(device) * 0.5))


  history = {'train':{'loss':[], 'driving':[], 'repulsive':[]}}

  print('-------------------Start training------------------')

  for i in range(config.n_epochs):
    for batch_x, batch_y in train_dataloader:
      batch_x = batch_x.reshape(len(batch_x), -1).to(device)
      batch_y = batch_y.to(device)

      if config.adversarial:  #adversarial examples
        samples = batch_x.requires_grad_(True)
        logits, pred = ensemble.forward(samples, W)
        nll_loss = torch.stack([F.nll_loss(F.log_softmax(p, -1), batch_y) for p in pred])
        adv_grad = torch.autograd.grad(nll_loss.sum(), samples)[0]
        adv_x = samples + config.eps * torch.sgn(adv_grad)
        batch_x = torch.vstack([batch_x, adv_x]).detach()
        batch_y = torch.concatenate([batch_y, batch_y])

      optimizer.zero_grad()
      W_ = W.detach().requires_grad_(True)
      logits, pred = ensemble.forward(batch_x, W_)

      #compute log likelihood
      nll_loss = torch.stack([F.nll_loss(F.log_softmax(p, -1), batch_y) for p in pred])
      nll_loss *= num_train 
      history['train']['loss'].append(torch.mean(nll_loss).item())

      optimizer.zero_grad()
      #add log prior and repulsive term
      if not config.functional:
        log_priors = []
        for ind, p in enumerate(W_):
            log_priors.append(priors[ind].log_prob(p).sum())

        nll_loss = torch.add(-torch.stack(log_priors), nll_loss)
        neg_score_func = torch.autograd.grad(nll_loss.sum(), W_)[0]

        #compute repulsive term
        grad_density = repulsive_term(param=W_, kernel=kernel, method=config.method, num_models=W.shape[0])
        W.grad = config.anneal * neg_score_func + grad_density
      else:
        neg_score_func = torch.autograd.grad(nll_loss.sum(), pred, retain_graph=True)[0].view(W.shape[0],-1)

        #gradient functional prior
        pred = pred.view(W.shape[0],-1)
        w_prior = torch.stack([prior.sample() for prior in priors], 0)
        prior_pred = ensemble.forward(batch_x, w_prior)[1].reshape(W.shape[0],-1)
        grad_prior = ssge.compute_score_gradients(pred, prior_pred)
        neg_score_func -= grad_prior

        #compute repulsive term

        #pred_add = (ensemble.forward(X_add,W_)[1]).view(W.shape[0],-1)
        pred_add = pred
        grad_density = repulsive_term(param=pred_add, kernel=kernel, method=config.method, num_models=W.shape[0])

        #gradient of log posterior
        phi = config.anneal * neg_score_func + grad_density
        W.grad = torch.autograd.grad(pred, W_, grad_outputs=phi,retain_graph=False)[0]

      optimizer.step()
      driving, repulsive = -neg_score_func*config.anneal, -grad_density
      history['train']['driving'].append(torch.mean(torch.abs(driving)).item())
      history['train']['repulsive'].append(torch.mean(torch.abs(repulsive)).item())

    scheduler.step()

    print('Train epoch:', i, ' train loss:', np.mean(history['train']['loss'][-len(train_dataloader):]),
          'driving:', history['train']['driving'][-1],
          'repulsive:', history['train']['repulsive'][-1])
  return history

