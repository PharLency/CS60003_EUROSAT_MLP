import numpy as np


class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, model):
        model.W1 -= self.lr * model.dW1
        model.b1 -= self.lr * model.db1
        model.W2 -= self.lr * model.dW2
        model.b2 -= self.lr * model.db2
        model.W3 -= self.lr * model.dW3
        model.b3 -= self.lr * model.db3


class StepLRScheduler:
    def __init__(self, optimizer, step_size=10, gamma=0.5):
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.initial_lr = optimizer.lr

    def step(self, epoch):
        self.optimizer.lr = self.initial_lr * (self.gamma ** (epoch // self.step_size))
        return self.optimizer.lr