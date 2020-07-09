from gates import gate_in1out1, gate_in2out1
from data import (
    dataset_and, 
    dataset_nand,
    dataset_nor,
    dataset_or,
    dataset_xor,
    dataset_not)
from torch.utils.data import DataLoader
import torch.nn as nn
import torch


def init():
    gates = [
        gate_in2out1.GateIn2Out1(),
        gate_in2out1.GateIn2Out1(),
        gate_in2out1.GateIn2Out1(),
        gate_in2out1.GateIn2Out1(),
        gate_in2out1.GateIn2Out1(),
        gate_in1out1.GateIn1Out1()
    ]
    datasets = [
        dataset_and.DatasetAndGate(1000),
        dataset_nand.DatasetNandGate(1000),
        dataset_nor.DatasetNorGate(1000),
        dataset_or.DatasetOrGate(1000),
        dataset_xor.DatasetXorGate(1000),
        dataset_not.DatasetNotGate(1000)
    ]
    dataloaders = []
    for i in range(len(datasets)):
        dataloaders.append(
            DataLoader(datasets[i], batch_size=64, shuffle=True)
        )
    
    criterion = nn.MSELoss()
    optimizers = []
    for i in range(len(datasets)):
        optimizers.append(torch.optim.Adam(gates[i].parameters()))

    return {
        'models': gates,
        'dataloaders': dataloaders,
        'criterion': criterion,
        'optimizers': optimizers
    }


def train_epoch(epoch_idx, model, data_loader, criterion, optimizer):
    model.train()

    for iter_idx, (data) in enumerate(data_loader):
        optimizer.zero_grad()

        x = data[:][:len(data)-1]
        y = data[:][-1]

        y_pred = model(*x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

        y_pred[y_pred > 0.9] = 1
        y_pred[y_pred <= 0.1] = 0
        acc = (y_pred.eq(y).sum().item()) / (y_pred.nelement())

        if iter_idx % 100 == 0:
            print(type(data_loader.dataset), epoch_idx, '{}/{}'.format(iter_idx, len(data_loader)), loss.item(), acc)

        # if iter_idx == len(data_loader) - 1:
        #     print(data[0], y_pred[0])


def train_model(epoch_num, model, dataloader, criterion, optimizer):
    for i in range(epoch_num):
        train_epoch(i, model, dataloader, criterion, optimizer)
    torch.save(model.state_dict(), './{}.pth'.format(
        dataloader.dataset.__class__.__name__.split('.')[-1]
    ))


def train(models, dataloaders, criterion, optimizers):
    assert len(models) == len(dataloaders)

    for i in range(len(models)):
        train_model(1000, models[i], dataloaders[i], criterion, optimizers[i])


def main():
    config = init()
    train(**config)


if __name__ == "__main__":
    main()