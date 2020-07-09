import gates.gate
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
    names = [
        'and',
        'nand',
        'nor',
        'not',
        'or',
        'xor'
    ]
    name2path = {
        'and': torch.load('./ckpts/DatasetAndGate.pth'),
        'nand': torch.load('./ckpts/DatasetNandGate.pth'),
        'nor': torch.load('./ckpts/DatasetNorGate.pth'),
        'not': torch.load('./ckpts/DatasetNotGate.pth'),
        'or': torch.load('./ckpts/DatasetOrGate.pth'),
        'xor': torch.load('./ckpts/DatasetXorGate.pth'),
    }
    thegates = [gates.gate.Gate(name, name2path) for name in names]

    datasets = [
        dataset_and.DatasetAndGate(1000),
        dataset_nand.DatasetNandGate(1000),
        dataset_nor.DatasetNorGate(1000),
        dataset_not.DatasetNotGate(1000),
        dataset_or.DatasetOrGate(1000),
        dataset_xor.DatasetXorGate(1000)
    ]
    dataloaders = []
    for i in range(len(datasets)):
        dataloaders.append(
            DataLoader(datasets[i], batch_size=64, shuffle=False)
        )

    return {
        'models': thegates,
        'dataloaders': dataloaders
    }


def test_model(model, data_loader):
    model.eval()

    total_num = 0
    correct_num = 0
    for iter_idx, (data) in enumerate(data_loader):

        x = data[:][:len(data)-1]
        y = data[:][-1]

        y_pred = model(*x)

        y_pred[y_pred > 0.9] = 1
        y_pred[y_pred < 0.1] = 0

        correct_num += y_pred.eq(y).sum().item()
        total_num += y_pred.nelement()

    return correct_num / total_num


def test(models, dataloaders):
    assert len(models) == len(dataloaders)

    for i in range(len(models)):
        acc = test_model(models[i], dataloaders[i])
        print(models[i].name, acc)


def main():
    config = init()
    test(**config)


if __name__ == "__main__":
    main()