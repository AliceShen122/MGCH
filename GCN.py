# coding: utf-8
import torch
import torch.nn as nn
from GAT import GAT
from torch.autograd import Variable


class GCN(nn.Module):
    def __init__(self, n_input, n_output, n_class, batch_size=20):
        super(GCN, self).__init__()

        # self.anchor_affnty = anchor_affnty
        self.batch_size = batch_size

        self.gconv1 = nn.Linear(n_input, 2048)
        self.BN1 = nn.BatchNorm1d(2048)
        self.act1 = nn.ReLU()  # g(x) = max(0, x)

        # self.x_1 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, padding=0, groups=1)
        # self.x_2 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, padding=0, groups=1)
        # self.x_3 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, padding=0, groups=1)
        # self.conv3 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1)

        # self.gconv2 = nn.Linear(2048, 2048)
        # self.BN2 = nn.BatchNorm1d(2048)
        # self.act2 = nn.ReLU()
        self.gat = GAT(nfeat=2048, nhid=8, nclass=256, dropout=0.6, nheads=2, alpha=0.2)

        self.gconv3 = nn.Linear(256, n_output)
        self.BN3 = nn.BatchNorm1d(n_output)
        self.act3 = nn.Tanh()

        # self.gconv4 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, padding=0, groups=1)
        # self.gconv5 = nn.Linear(54, n_output)

        self.fc = nn.Linear(n_output, n_class)

    # def forward(self, x, in_affnty_1, out_affnty_1, anchor_affnty_1, s):  # (120,4608) ; (800,120) ; (120,800) ; (800,800) ; 40
    #
    #     out = self.gconv1(x)  # (120,2048)
    #     out = in_affnty_1.mm(out)  # (800,2048)
    #     out = self.BN1(out)  # (800,2048)
    #     out = self.act1(out)  # (800,2408)
    #
    #     # block 2
    #     # out = self.gconv2(out)  # (800,2408)
    #     # out = anchor_affnty.mm(out)  # (800,2408)
    #     # out = self.BN2(out)  # (800,2408)
    #     # out = self.act2(out)  # (800,2408)
    #     out = self.gat(out, anchor_affnty_1)  # (800,38)
    #
    #     # block 3
    #     out = self.gconv3(out)  # (800,16)
    #     out = out_affnty_1.mm(out)  # (120,16)
    #     out = self.BN3(out)  # (120,16)
    #     out = self.act3(out)  # (120,16)
    #
    #     if self.training is False:
    #         return out
    #
    #     out_masked = out[:s, :]  # (100,16)
    #     pred = self.fc(out_masked)  # (100,38)
    #
    #     return out, pred  # (120,64) (40,80)

    def forward(self, x, in_affnty_1, out_affnty_1, anchor_affnty_1,
                s):  # (120,4608) ; (800,120) ; (120,800) ; (800,800) ; 40
        # in_affnty = torch.stack((in_affnty_1, in_affnty_2, in_affnty_3))  # (3,800,120)
        # out_affnty = torch.stack((out_affnty_1, out_affnty_2, out_affnty_3))  # (3,120,800)
        # anchor_affnty = torch.stack((anchor_affnty_1, anchor_affnty_2, anchor_affnty_3))  # (3,800,800)
        #
        # in_affnty = Variable(torch.unsqueeze(in_affnty, dim=0).float(), requires_grad=False)  # (1,3,800,120)
        # out_affnty = Variable(torch.unsqueeze(out_affnty, dim=0).float(), requires_grad=False)  # (1,3,120,800)
        # anchor_affnty = Variable(torch.unsqueeze(anchor_affnty, dim=0).float(), requires_grad=False)  # (1,3,800,800)
        #
        # in_affnty = self.x_1(in_affnty)  # (1,1,800,120)
        # in_affnty = torch.squeeze(in_affnty)  # (800,120)
        # out_affnty = self.x_2(out_affnty)  # (1,1,120,800)
        # out_affnty = torch.squeeze(out_affnty)  # (120,800)
        # anchor_affnty = self.x_3(anchor_affnty)  # (1,1,800,800)
        # anchor_affnty = torch.squeeze(anchor_affnty)  # (800,800)

        out_1 = self.gconv1(x)  # (120,2048)
        out_1 = in_affnty_1.mm(out_1)  # (800,2048)
        out_1 = self.BN1(out_1)  # (800,2048)
        out_1 = self.act1(out_1)  # (800,2408)

        # block 2
        # out = self.gconv2(out)  # (800,2408)
        # out = anchor_affnty.mm(out)  # (800,2408)
        # out = self.BN2(out)  # (800,2408)
        # out = self.act2(out)  # (800,2408)
        out_1 = self.gat(out_1, anchor_affnty_1)  # (800,38)

        # block 3
        out_1 = self.gconv3(out_1)  # (800,16)
        out_1 = out_affnty_1.mm(out_1)  # (120,16)
        out_1 = self.BN3(out_1)  # (120,16)
        out = self.act3(out_1)  # (120,16)
        torch.cuda.empty_cache()
        ##################################################
        # out_2 = self.gconv1(x)  # (120,2048)
        # out_2 = in_affnty_2.mm(out_2)  # (800,2048)
        # out_2 = self.BN1(out_2)  # (800,2048)
        # out_2 = self.act1(out_2)  # (800,2408)
        #
        # out_2 = self.gat(out_2, anchor_affnty_2)  # (800,38)
        #
        # # block 3
        # out_2 = self.gconv3(out_2)  # (800,16)
        # out_2 = out_affnty_2.mm(out_2)  # (120,16)
        # out_2 = self.BN3(out_2)  # (120,16)
        # out_2 = self.act3(out_2)  # (120,16)
        # torch.cuda.empty_cache()
        ########################################################
        # out_3 = self.gconv1(x)  # (120,2048)
        # out_3 = in_affnty_3.mm(out_3)  # (800,2048)
        # out_3 = self.BN1(out_3)  # (800,2048)
        # out_3 = self.act1(out_3)  # (800,2408)
        # torch.cuda.empty_cache()
        # out_3 = self.gat(out_3, anchor_affnty_3)  # (800,38)
        #
        # # block 3
        # out_3 = self.gconv3(out_3)  # (800,16)
        # out_3 = out_affnty_3.mm(out_3)  # (120,16)
        # out_3 = self.BN3(out_3)  # (120,16)
        # out_3 = self.act3(out_3)  # (120,16)
        # torch.cuda.empty_cache()
        ########################################################
        # out_4 = self.gconv1(x)  # (120,2048)
        # out_4 = in_affnty_4.mm(out_4)  # (800,2048)
        # out_4 = self.BN1(out_4)  # (800,2048)
        # out_4 = self.act1(out_4)  # (800,2408)
        # torch.cuda.empty_cache()
        # out_4 = self.gat(out_4, anchor_affnty_4)  # (800,38)
        #
        # # block 3
        # out_4 = self.gconv3(out_4)  # (800,16)
        # out_4 = out_affnty_4.mm(out_4)  # (120,16)
        # out_4 = self.BN3(out_4)  # (120,16)
        # out_4 = self.act3(out_4)  # (120,16)
        # torch.cuda.empty_cache()
        ########################################################
        # out_5 = self.gconv1(x)  # (120,2048)
        # out_5 = in_affnty_5.mm(out_5)  # (800,2048)
        # out_5 = self.BN1(out_5)  # (800,2048)
        # out_5 = self.act1(out_5)  # (800,2408)
        # torch.cuda.empty_cache()
        # out_5 = self.gat(out_5, anchor_affnty_5)  # (800,38)
        #
        # # block 3
        # out_5 = self.gconv3(out_5)  # (800,16)
        # out_5 = out_affnty_5.mm(out_5)  # (120,16)
        # out_5 = self.BN3(out_5)  # (120,16)
        # out_5 = self.act3(out_5)  # (120,16)
        # torch.cuda.empty_cache()
        ########################################################
        # out_6 = self.gconv1(x)  # (120,2048)
        # out_6 = in_affnty_6.mm(out_6)  # (800,2048)
        # out_6 = self.BN1(out_6)  # (800,2048)
        # out_6 = self.act1(out_6)  # (800,2408)
        # torch.cuda.empty_cache()
        # out_6 = self.gat(out_6, anchor_affnty_6)  # (800,38)
        #
        # # block 3
        # out_6 = self.gconv3(out_6)  # (800,16)
        # out_6 = out_affnty_6.mm(out_6)  # (120,16)
        # out_6 = self.BN3(out_6)  # (120,16)
        # out_6 = self.act3(out_6)  # (120,16)
        # torch.cuda.empty_cache()
        ########################################################
        # out = torch.cat((out_1, out_2), dim=1)  # (120,54*3)

        # out = torch.stack((out_1, out_2, out_3))  # (3,120,54)
        # out = Variable(torch.unsqueeze(out, dim=0).float(), requires_grad=False)  # (1,3,120,54)
        # out = self.gconv4(out)  # (1,1,120,54)
        # out = torch.squeeze(out)  # (120,54)

        # out = self.gconv5(out)  # (120,nbits)

        if self.training is False:
            return out

        out_masked = out[:s, :]  # (100,16)
        pred = self.fc(out_masked)  # (100,38)

        return out, pred  # (120,64) (40,80)
