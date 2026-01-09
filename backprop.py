import torch
import torch.optim as optim
import torch.nn as nn

def train_step(model, data, target, optimizer):
    """
    Performs a single training step.
    """
    optimizer.zero_grad()
    output = model(data)
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
    return loss.item()

def create_optimizer(model, learning_rate=0.001):
    """
    Creates an Adam optimizer for the model.
    """
    return optim.Adam(model.parameters(), lr=learning_rate)
