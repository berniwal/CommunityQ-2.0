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
from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
# from pytorch_lightning.loggers import WandbLogger

from transformers import BertTokenizer, BertModel
from transformers import DistilBertTokenizer, DistilBertModel, DistilBertForSequenceClassification


def visualize_histograms(dataset, vis_folder, output_name):
    if not os.path.exists(vis_folder):
        os.mkdir(vis_folder)

    if 'QuestionText' in dataset:
        dataset = dataset.drop('QuestionText', inplace=False, axis=1)
    height = int(np.sqrt(dataset.shape[1]))
    width = (dataset.shape[1] // height) + 1
    fig, axs = plt.subplots(height, width, figsize=(20, 8.27), tight_layout=True)
    for feature_index in range(dataset.shape[1]):
        current_axis = axs[feature_index // width, feature_index % width]
        current_data = dataset.iloc[:, feature_index].sample(1000)
        feature_name = current_data.name
        current_axis.hist(current_data)
        current_axis.set_title(feature_name, size=8)
        # current_axis.set_yscale('log')
    plt.savefig(os.path.join(vis_folder, output_name))


class QuestionDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_directory, output_features, input_features_numerical=None,
                 input_features_categorical=None,
                 input_features_text=None,
                 visualization_folder='./visualizations'):
        self.dataset = pd.read_csv(dataset_directory)
        self.y_raw = self.dataset[output_features]

        visualize_histograms(self.dataset, visualization_folder, output_name='dataset.png')

        self.x_num = None
        if input_features_numerical is not None:
            self.x_num_raw = self.dataset[input_features_numerical]
            self.mean_x = self.x_num_raw.mean()
            self.std_x = self.x_num_raw.std()
            visualize_histograms(self.x_num_raw, visualization_folder, output_name='unnormalized.png')
            self.x_num = (self.x_num_raw - self.mean_x) / self.std_x
            # self.x_num = self.x_num_raw / self.x_num_raw.max()
            visualize_histograms(self.x_num, visualization_folder, output_name='normalized.png')
            self.x_num = self.x_num.to_numpy()

        self.x_cat = None
        if input_features_categorical is not None:
            self.x_cat = np.zeros((self.dataset.shape[0], 0))
            for categorical_feature in input_features_categorical:
                if categorical_feature == 'BrandId':
                    mask = self.x_num_raw['VisitsLastYear'] > 20000
                    self.dataset[categorical_feature] = self.dataset[categorical_feature].where(mask, other=-1)
                current_dummy = pd.get_dummies(self.dataset[categorical_feature]).to_numpy()
                self.x_cat = np.concatenate((self.x_cat, current_dummy), axis=1)

        self.x_text = self.dataset[input_features_text].to_numpy().squeeze()

        '''output_file = './data/questions.txt'
        with open(output_file, 'a') as file:
            for test in tqdm(self.x_text):
                if '\n' in test:
                    test = test.replace('\n', '')
                file.write(test + '\n')'''

        # self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-german-cased', use_fast=True)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased', use_fast=True)
        self.MAX_LEN = 512

        # test_text = self.x_text[1]
        # test_tokens = self.tokenizer.encode_plus(test_text, add_special_tokens=True, max_length=self.MAX_LEN, truncation=True)
        # test_back_text = [list(self.tokenizer.vocab)[int(x)] for x in test_tokens['input_ids']]

        self.x = np.concatenate((self.x_num, self.x_cat), axis=1)
        self.y = self.y_raw.to_numpy()

    def __len__(self):
        return int(self.x.shape[0])

    def __getitem__(self, index):
        x_num = torch.tensor(self.x[index], dtype=torch.float32)

        x_text = self.x_text[index]
        inputs = self.tokenizer.encode_plus(x_text, add_special_tokens=True, max_length=self.MAX_LEN, truncation=True)
        input_ids = inputs["input_ids"]
        pad_id = self.tokenizer.pad_token_id
        padding_length = self.MAX_LEN - len(input_ids)
        input_ids_padded = input_ids + ([pad_id] * padding_length)
        input_ids_padded = torch.tensor(input_ids_padded)

        y = torch.tensor(self.y[index])
        return (x_num, input_ids_padded), y


class MLP_Block(nn.Module):
    def __init__(self, input_dimension: int = 1024, output_dimension: int = 1024, bias: bool = True,
                 dropout: bool = False):
        super().__init__()
        self.fc = nn.Linear(input_dimension, output_dimension, bias=bias)
        self.bn = nn.BatchNorm1d(output_dimension)
        self.act = nn.ReLU()
        self.dropout = None
        if dropout:
            self.dropout = nn.Dropout(0.1, inplace=True)

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        x = self.act(x)
        if self.dropout is not None: x = self.dropout(x)
        return x


class SimpleQuestionAnswerer(nn.Module):
    def __init__(self, input_dimension: int = 2, hidden_dimension: int = 1024, num_layers: int = 4,
                 num_classes: int = 2, bias: bool = True):
        super().__init__()

        self.fc_in = MLP_Block(input_dimension, hidden_dimension, bias=bias)
        self.mlp_hidden = nn.ModuleList([
            MLP_Block(hidden_dimension, hidden_dimension, bias=bias) for i in range(num_layers)
        ])
        self.fc_out = nn.Linear(hidden_dimension, num_classes, bias=bias)

    def forward(self, x):
        x = F.relu(self.fc_in(x))
        for mlp in self.mlp_hidden:
            x = mlp(x)

        x = self.fc_out(x)
        return x


class QuestionAnswerer(pl.LightningModule):
    def __init__(self, input_dimension: int = 2, hidden_dimension: int = 1024, num_layers: int = 4,
                 num_classes: int = 2, bias: bool = True, mean: pd.core.series.Series = None,
                 std: pd.core.series.Series = None, nlp_backbone: bool = True):
        super().__init__()
        self.mean = mean
        self.std = std
        self.nlp_backbone = nlp_backbone
        if self.nlp_backbone:
            # self.BERT_backbone = DistilBertForSequenceClassification.from_pretrained('distilbert-base-german-cased', num_labels=num_classes)
            self.BERT_backbone = BertModel.from_pretrained('bert-base-german-cased', num_labels=num_classes)
            # self.BERT_backbone = DistilBertModel.from_pretrained('distilbert-base-german-cased', num_labels=num_classes)
            self.backbone = SimpleQuestionAnswerer(input_dimension + 768, hidden_dimension, num_layers, num_classes, bias)
        else:
            self.backbone = SimpleQuestionAnswerer(input_dimension, hidden_dimension, num_layers, num_classes, bias)
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.y_pred = []
        self.y_gt = []

    def forward(self, x):
        x_num, x_text = x
        if self.nlp_backbone:
            # logits = self.BERT_backbone(x_text).logits
            text_hidden_state = self.BERT_backbone(x_text).last_hidden_state
            text_hidden_state = text_hidden_state[:, 0]
            x_final = torch.cat((x_num, text_hidden_state), 1)
            logits = self.backbone(x_final)
        else:
            logits = self.backbone(x_num)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.cross_entropy_loss(logits, y[:, -1])
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.cross_entropy_loss(logits, y[:, -1])
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        predictions = logits.argmax(dim=1).numpy()
        y_pred.extend(predictions)
        y_gt.extend(y[:, -1])
        import pdb
        pdb.set_trace()
        loss = self.cross_entropy_loss(logits, y[:, -1])
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=5e-05)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               factor=0.1,
                                                               patience=5,
                                                               min_lr=1e-07,
                                                               verbose=1)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}


def main(args):
    input_features_numerical = ['VerifiedBuyers', 'VisitsLastYear',
                                'QuestionTextLength', 'ProductLifecycleDays', 'ProductQuestions',
                                'ProductQuestionsFractionAnswered',
                                'ProductQuestionsFractionAnsweredWithinWeek',
                                'ProductQuestionsHaveNonEmployeeAnswers',
                                'ProductQuestionsHaveNonEmployeeAnswersWithinWeek',
                                'ProductQuestionsNonEmployeeAnswers',
                                'ProductQuestionsNonEmployeeAnswersWithinWeek',
                                'VerifiedBuyersLastMonth', 'VerifiedBuyersLastYear',
                                'VerifiedBuyersWithLanguage',
                                'VerifiedBuyersWithLanguageAnswersTotalScore',
                                'VerifiedBuyersWithLanguageBestAnswers',
                                'VerifiedBuyersWithLanguageHaveAnswered',
                                'VerifiedBuyersWithLanguageLastMonth',
                                'VerifiedBuyersWithLanguageLastYear',
                                'VerifiedBuyersWithLanguageTotalAnswers',
                                'VerifiedBuyersWithLanguageTotalAnswersAsVerifiedBuyers',
                                'VisitsLastMonth']
    input_features_categorical = ['ProductGroup1Id', 'BrandId', 'CategoryManagementTeamBudgetingGroupId',
                                  'CategoryManagementTeamId']
    input_features_text = ['QuestionText']
    output_features = ['IsQuestionForCommunity']

    dataset = QuestionDataset(dataset_directory=args['dataset'],
                              input_features_numerical=input_features_numerical,
                              input_features_categorical=input_features_categorical,
                              input_features_text=input_features_text,
                              output_features=output_features,
                              visualization_folder=args['vis_save_path']
                              )

    dataset_samples = dataset.__len__()
    train_samples = int(0.8 * dataset_samples)
    val_samples = int(0.5 * (dataset_samples - train_samples))
    test_samples = dataset_samples - train_samples - val_samples
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_samples, val_samples, test_samples])

    train_dataloader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True, num_workers=10)
    val_dataloader = DataLoader(val_dataset, batch_size=args['batch_size'], num_workers=10)
    test_dataloader = DataLoader(test_dataset, batch_size=args['batch_size'], num_workers=10)

    net = QuestionAnswerer(input_dimension=dataset.x.shape[1],
                           hidden_dimension=args['hidden_dimension'],
                           num_layers=args['num_layers'],
                           num_classes=2,
                           bias=args['bias'],
                           mean=dataset.mean_x,
                           std=dataset.std_x,
                           )

    print(net)

    # wandb_logger = WandbLogger(name='Test-Run', project='Digitec')
    # wandb_logger.watch(net)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        dirpath='./checkpoints/{}'.format(args['experiment_id']),
        save_last=True,
        save_top_k=5,
        save_weights_only=False,
        filename='model-{epoch:03d}-{val_loss:.4f}'
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min'
    )

    trainer = pl.Trainer(gpus=1,
                         callbacks=[checkpoint_callback, early_stop_callback],
                         # logger=[wandb_logger],
                         # overfit_batches=5
                         )

    if not args['only_test']:
        trainer.fit(net, train_dataloader=train_dataloader, val_dataloaders=val_dataloader)
        trainer.test(test_dataloaders=test_dataloader)
    else:
        if args['model_path'] is not None:
            state_dict = torch.load(config['model_path'])['state_dict']
            net.load_state_dict(state_dict)
            import pdb
            pdb.set_trace()
            trainer.test(net, test_dataloader=test_dataloader)
        else:
            print('Please specify model_path variable, where to load the Model.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Input Arguments')
    parser.add_argument('--experiment_id', default='default', type=str, help='Experiment Id to choose')
    parser.add_argument('--dataset', default='./data/juniorMLE_dataset.csv', type=str, help='Input Path to Dataset')
    parser.add_argument('--bias', default=True, type=bool, help='Bias')
    parser.add_argument('--only_test', default=False, type=bool, help='If only testing should be performed')
    parser.add_argument('--epochs', default=10, type=int, help='Epochs to train')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch Size to use')
    parser.add_argument('--num_layers', default=2, type=int, help='Number of Layers of MLP')
    parser.add_argument('--hidden_dimension', default=2048, type=int, help='Hidden Dimension')
    parser.add_argument('--model_path', default=None, type=str, help='Path to load model for testing')
    parser.add_argument('--vis_save_path', default='./visualizations', type=str, help='Path to save visualizations')
    args = parser.parse_args()
    args = vars(args)
    main(args)
