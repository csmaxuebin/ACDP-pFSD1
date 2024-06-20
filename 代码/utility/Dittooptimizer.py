import torch
from torch.optim import Optimizer

class PerturbedGradientDescent(Optimizer):
    def __init__(self, params, lr=0.1, mu=0.1):
        default = dict(lr=lr, mu=mu)
        super().__init__(params, default)
    @torch.no_grad()
    def step(slef,args,net,net_glob):
        for w, w_t in zip(net.parameters(), net_glob.parameters()):
            term = w.grad.data + args.mu * (w.data - w_t.data)
            w.data.add_(term,alpha=-args.lr)