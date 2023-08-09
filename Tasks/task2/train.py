'''
AI Fashion Coordinator
(Baseline For Fashion-How Challenge)

MIT License

Copyright (C) 2022, Integrated Intelligence Research Section, ETRI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Update: 2022.04.20.
'''
from dataset import ETRIDataset_color
from networks import *

import pandas as pd
import os
import argparse
import time

import torch
import torch.utils.data
import torch.utils.data.distributed


parser = argparse.ArgumentParser()
parser.add_argument("--version", type=str, default='Baseline_ResNet_color')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--lr', default=0.0001, type=float, metavar='N',
                    help='learning rate')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 64), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')

a = parser.parse_args()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    """ The main function for model training. """
    if os.path.exists('models') is False:
        os.makedirs('models')

    save_path = 'models/' + time.strftime('%m-%d', time.localtime()) + '/' + a.version
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    net = Baseline_ResNet_color().to(DEVICE)

    df = pd.read_csv('../Dataset/task2/info_etri20_color_train.csv')
    train_dataset = ETRIDataset_color(df, base_path='../Dataset/task2/Train/')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=a.batch_size, shuffle=True, num_workers=0)

    optimizer = torch.optim.Adam(net.parameters(), lr=a.lr)
    criterion = nn.CrossEntropyLoss().to(DEVICE)

    total_step = len(train_dataloader)
    step = 0
    t0 = time.time()

    for epoch in range(a.epochs):
        net.train()

        for i, sample in enumerate(train_dataloader):
            optimizer.zero_grad()
            step += 1
            for key in sample:
                sample[key] = sample[key].to(DEVICE)

            out = net(sample)

            loss = criterion(out, sample['color_label'])

            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Time : {:2.3f}'
                      .format(epoch + 1, a.epochs, i + 1, total_step, loss.item(), time.time() - t0))

                t0 = time.time()

        if ((epoch + 1) % 10 == 0):
            a.lr *= 0.9
            optimizer = torch.optim.Adam(net.parameters(), lr=a.lr)
            print("learning rate is decayed")


        if ((epoch + 1) % 20 == 0):
            print('Saving Model....')
            torch.save(net.state_dict(), save_path + '/model_' + str(epoch + 1) + '.pkl')
            print('OK.')



if __name__ == '__main__':
    main()

