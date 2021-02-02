import argparse
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader


class QuestionDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_directory, input_features, output_features):
        self.dataset = pd.read_csv(dataset_directory)
        self.x_raw = self.dataset[input_features]
        self.y_raw = self.dataset[output_features]

        self.mean_x = self.x_raw.mean()
        self.std_x = self.x_raw.std()

        self.x = (self.x_raw - self.mean_x) / self.std_x

        self.x = self.x.to_numpy()
        self.y = self.y_raw.to_numpy()

    def __len__(self):
        return int(self.x.shape[0])

    def __getitem__(self, index):
        x = torch.tensor(self.x[index], dtype=torch.float32)
        y = torch.tensor(self.y[index])
        return x, y


class SimpleQuestionAnswerer(nn.Module):
    def __init__(self, input_dimension: int = 2, hidden_dimension: int = 1024, num_layers: int = 4,
                 num_classes: int = 2, bias: bool = True, mean: pd.core.series.Series = None,
                 std: pd.core.series.Series = None):
        super().__init__()
        self.mean = mean
        self.std = std

        self.fc_in = nn.Linear(input_dimension, hidden_dimension, bias=bias)
        self.fc_hidden = nn.ModuleList([
            nn.Linear(hidden_dimension, hidden_dimension, bias=bias) for i in range(num_layers)
        ])
        self.fc_out = nn.Linear(hidden_dimension, num_classes, bias=bias)

    def forward(self, x):
        x = F.relu(self.fc_in(x))
        for fc in self.fc_hidden:
            x = F.relu(fc(x))
        x = self.fc_out(x)
        return x


def create_scatter_plot(x, y, input_features, title, output_path):
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel(input_features[0])
    ax.set_ylabel(input_features[1])
    scatter = ax.scatter(x[:, 0], x[:, 1], c=y)
    legend = ax.legend(*scatter.legend_elements(), title='IsQuestionForCommunity')
    ax.add_artist(legend)
    plt.savefig(output_path)
    plt.clf()


def main(args):
    input_features = ['VisitsLastYear', 'QuestionTextLength']
    output_features = ['IsQuestionForCommunity']

    train_dataset = QuestionDataset(dataset_directory=args['dataset'],
                                    input_features=input_features,
                                    output_features=output_features,
                                    )

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    visualization_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False)

    net = SimpleQuestionAnswerer(input_dimension=len(input_features),
                                 hidden_dimension=args['hidden_dimension'],
                                 num_layers=args['num_layers'],
                                 num_classes=2,
                                 bias=args['bias'],
                                 mean=train_dataset.mean_x,
                                 std=train_dataset.std_x,
                                 )
    print(net)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(args['epochs']):
        pbar = tqdm(train_dataloader, desc='Epoch {}/{}'.format(epoch, args['epochs']))
        losses = []
        for idx, data in enumerate(pbar):
            x, y = data
            optimizer.zero_grad()
            out = net(x)
            loss = criterion(out, y[:, -1])
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if idx % 10 == 0:
                pbar.set_postfix({'Loss': np.mean(losses)})
        print('{} Epoch - Loss {}'.format(epoch, np.mean(losses)))

    torch.save(net, os.path.join(args['model_save_path'], 'final_model.pt'))
    print('Finished Training')

    print('Start Visualization')
    net.eval()
    pbar = tqdm(visualization_dataloader, desc='Visualization')
    y_pred = []
    for x, y in pbar:
        out = net(x)
        predictions = out.argmax(dim=1).numpy()
        y_pred.extend(predictions)

    x = train_dataset.x_raw.to_numpy()
    y = train_dataset.y_raw.to_numpy()

    output_directory = args['vis_save_path']
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    output_path = os.path.join(output_directory, 'ground_truth.png')
    create_scatter_plot(x, y, input_features, 'Ground Truth', output_path)

    output_path = os.path.join(output_directory, 'prediction.png')
    create_scatter_plot(x, y_pred, input_features, 'Prediction', output_path)

    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    print('TN: {} FP: {} FN: {} TP: {}'.format(tn, fp, fn, tp))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Input Arguments')
    parser.add_argument('--dataset', default='./data/juniorMLE_dataset.csv', type=str, help='Input Path to Dataset')
    parser.add_argument('--bias', default=True, type=bool, help='Bias')
    parser.add_argument('--epochs', default=10, type=int, help='Epochs to train')
    parser.add_argument('--num_layers', default=1, type=int, help='Number of Layers of MLP')
    parser.add_argument('--hidden_dimension', default=128, type=int, help='Hidden Dimension')
    parser.add_argument('--model_save_path', default='./', type=str, help='Path to save model')
    parser.add_argument('--vis_save_path', default='./visualizations', type=str, help='Path to save visualizations')
    args = parser.parse_args()
    args = vars(args)
    main(args)
