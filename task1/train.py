'''
※ 로컬에서 학습을 수행하기 위한 코드입니다. 
   실제 제출에 사용할 추론코드는 task.ipynb를 사용합니다.
'''


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

import warnings
warnings.filterwarnings("ignore")

from datasets import *
import networks

import pandas as pd
import os
import argparse
import yaml
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.utils.data.distributed

import sys
import subprocess
subprocess.check_call(["sudo", sys.executable, "-m", "pip", "install", "python-dotenv"])
subprocess.check_call(["sudo", sys.executable, "-m", "pip", "install", "wandb"])
import wandb
from dotenv import load_dotenv

DATA_PATH = '/content/data-task1'
SAVE_PATH = '../../Models/task1'

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def main(config, do_eval, save_path):
    # -- set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # -- set up model save path   
    if os.path.exists(SAVE_PATH) is False:
        os.makedirs(SAVE_PATH)

    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    
    # -- save config
    with open(f"{save_path}/config.yaml", 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    # -- get model
    net = networks.make(config['network']).to(device)
    print(">> network is prepared.")

    # -- get train dataset
    df_train = pd.read_csv(f'{DATA_PATH}/info_etri20_emotion_train.csv')
    train_dataset = Datasets()(df_train, config['img_size'], base_path=f'{DATA_PATH}/Train/')
    train_dataset = Augment(config['augment'])(train_dataset)
    train_dataset = Preprocess(config['preprocess'])(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    print(f">> train_dataset size : {len(train_dataset)}")
    
    # -- get valid dataset
    if do_eval:
        df_valid = pd.read_csv(f'{DATA_PATH}/info_etri20_emotion_valid.csv')
        valid_dataset = Datasets()(df_valid, config['img_size'], base_path=f'{DATA_PATH}/Valid/')
        valid_dataset = Augment(config['augment'])(valid_dataset)
        valid_dataset = Preprocess(config['preprocess'])(valid_dataset)
        valid_dataloader = DataLoader(
            valid_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)
        print(f">> valid_dataset size : {len(valid_dataset)}")  
    
    # -- get training components
    optimizer_enc = torch.optim.Adam(net.encoder.parameters(), lr=config['lr']['enc'])
    optimizer_dec1 = torch.optim.Adam(net.classifier1.parameters(), lr=config['lr']['dec'])
    optimizer_dec2 = torch.optim.Adam(net.classifier2.parameters(), lr=config['lr']['dec'])
    optimizer_dec3 = torch.optim.Adam(net.classifier3.parameters(), lr=config['lr']['dec'])
    optimizers = [optimizer_enc, optimizer_dec1, optimizer_dec2, optimizer_dec3]
    criterion = nn.CrossEntropyLoss().to(device)

    # -- wandb init
    load_dotenv(dotenv_path="wandb.env")
    WANDB_AUTH_KEY = os.getenv("WANDB_AUTH_KEY")
    wandb.login(key=WANDB_AUTH_KEY)
    wandb_name = save_path.split('/')[-1]
    wandb.init(
            entity="rond-7th",
            project="fashion-how",
            group=f"sub-task1",
            name=wandb_name
    )

    # -- start training
    for epoch in range(config['epochs']):
        # -- train step
        net.train()
        epoch_losses = [0, 0, 0, 0]
        for i, batch in enumerate(tqdm(train_dataloader, leave=False, desc='training')):
            
            for key in batch: batch[key] = batch[key].to(device)
            for optim in optimizers: optim.zero_grad()

            out_daily, out_gender, out_embel = net(batch['image'])
            loss_daily = criterion(out_daily, batch['daily'])
            loss_gender = criterion(out_gender, batch['gender'])
            loss_embel = criterion(out_embel, batch['embel'])
            loss = loss_daily + loss_gender + loss_embel
            loss.backward()

            for optim in optimizers: optim.step()
            for i, l in enumerate([loss, loss_daily, loss_gender, loss_embel]):
                epoch_losses[i] = (epoch_losses[i] * i + l.item()) / (i + 1)

        print(f"[{epoch+1:0>3}/{config['epochs']}]",
              f"loss={epoch_losses[0]:.4f}, loss_daily={epoch_losses[1]:.4f}, ",
              f"loss_gender={epoch_losses[2]:.4f}, loss_embel={epoch_losses[3]:.4f}")
        
        # -- valid step
        # if do_eval:
        #     with torch.no_grad() :
        #         net.eval()
        #         daily_acc, gender_acc, emb_acc = 0.0, 0.0, 0.0
        #         for i, batch in enumerate(tqdm(valid_dataloader, leave=False, desc='evaluating')):
        #             for key in batch: batch[key] = batch[key].to(device)
        #             daily_logit, gender_logit, embel_logit = net(batch['image'])


        if ((epoch + 1) % 20 == 0):
            torch.save(net.state_dict(), save_path + '/model_' + str(epoch + 1) + '.pkl')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str)
    parser.add_argument("--eval", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2023)
    args = parser.parse_args()

    # -- set up eval
    do_eval = args.eval

    # -- set up seed
    seed_everything(args.seed)

    # -- get config
    with open(args.cfg, 'r') as f:
        config = yaml.load(f, Loader = yaml.FullLoader)

    save_path = f"{SAVE_PATH}/{args.cfg.split('/')[-1][:-len('.yaml')]}"
    
    main(config, do_eval, save_path)