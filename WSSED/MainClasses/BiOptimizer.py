import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


def _concat(xs):
  return torch.cat([x.view(-1) for x in xs])


class BiOptimizer(object):
  def __init__(self, model, args):
    self.network_momentum = 0.9
    self.network_weight_decay = 3e-4
    self.model = model
    self.optimizer = torch.optim.Adam(self.model.hi_pool.parameters(),
        lr=args.eta, betas=(0.5, 0.999), weight_decay=1e-3)

  def _get_step_model(self, input, target, eta, network_optimizer):
    loss = self.model._loss(input, target)
    theta = _concat(self.model.crnn.parameters()).data
    try:
      moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.crnn.parameters()).mul_(self.network_momentum)
    except:
      moment = torch.zeros_like(theta)
    dtheta = _concat(torch.autograd.grad(loss, self.model.crnn.parameters())).data + self.network_weight_decay * theta
    base_model_temp = self._construct_model_from_theta(theta.sub(other=moment + dtheta, alpha=eta))
    return base_model_temp

  def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer):
    self.optimizer.zero_grad()
    self._backward_step(input_train, target_train, input_valid, target_valid, eta, network_optimizer)
    self.optimizer.step()

  def _backward_step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer):
    step_model = self._get_step_model(input_train, target_train, eta, network_optimizer)
    base_loss = step_model._loss(input_valid, target_valid)

    base_loss.backward()
    dalpha = [v.grad for v in step_model.hi_pool.parameters()] # alpha(hi pool w)
    vector = [v.grad.data for v in step_model.crnn.parameters()] # w'
    implicit_grads = self._hessian_vector_product(vector, input_train, target_train)

    for g, ig in zip(dalpha, implicit_grads):
      g.data.sub_(eta, ig.data)

    for v, g in zip(self.model.hi_pool.parameters(), dalpha):
      if v.grad is None:
        v.grad = Variable(g.data)
      else:
        v.grad.data.copy_(g.data)

  def _construct_model_from_theta(self, theta):
    model_new = self.model._new()
    model_dict = self.model.crnn.state_dict()

    params, offset = {}, 0
    for k, v in self.model.crnn.named_parameters():
      v_length = np.prod(v.size())
      params[k] = theta[offset: offset+v_length].view(v.size())
      offset += v_length

    assert offset == len(theta)
    model_dict.update(params)
    model_new.crnn.load_state_dict(model_dict)
    return model_new.cuda()

  def _hessian_vector_product(self, vector, input, target, r=1e-2):
    R = r / _concat(vector).norm()
    for p, v in zip(self.model.crnn.parameters(), vector):
      p.data.add_(R, v)
    loss = self.model._loss(input, target)
    grads_p = torch.autograd.grad(loss, self.model.hi_pool.parameters())

    for p, v in zip(self.model.parameters(), vector):
      p.data.sub_(2*R, v)
    loss = self.model._loss(input, target)
    grads_n = torch.autograd.grad(loss, self.model.hi_pool.parameters())

    for p, v in zip(self.model.crnn.parameters(), vector):
      p.data.add_(R, v)

    return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]

