import math
from typing import List, Union
import hydra
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.cuda.amp import autocast
from torch.nn import CTCLoss
from deepspeech_pytorch.real_dtw import compute_dtw

from deepspeech_pytorch.configs.train_config import DTWDataConfig

from deepspeech_pytorch.configs.train_config import (
    SpectConfig,
    BiDirectionalConfig,
    OptimConfig,
    AdamConfig,
    SGDConfig,
    UniDirectionalConfig,
)
from deepspeech_pytorch.loss import DTWLosslabels, DTWLosswithoutlabels
from deepspeech_pytorch.gauss import gaussrep


class SequenceWise(nn.Module):
    def __init__(self, module):
        """
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        :param module: Module to apply input to.
        """
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        t, n = x.size(0), x.size(1)
        x = x.view(t * n, -1)
        x = self.module(x)
        x = x.view(t, n, -1)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + " (\n"
        tmpstr += self.module.__repr__()
        tmpstr += ")"
        return tmpstr


class MaskConv(nn.Module):
    def __init__(self, seq_module):
        """
        Adds padding to the output of the module based on the given lengths. This is to ensure that the
        results of the model do not change when batch sizes change during inference.
        Input needs to be in the shape of (BxCxDxT)
        :param seq_module: The sequential module containing the conv stack.
        """
        super(MaskConv, self).__init__()
        self.seq_module = seq_module

    def forward(self, x, lengths):
        """
        :param x: The input of size BxCxDxT
        :param lengths: The actual length of each sequence in the batch
        :return: Masked output from the module
        """
        for module in self.seq_module:
            x = module(x)
            mask = torch.BoolTensor(x.size()).fill_(0)
            if x.is_cuda:
                mask = mask.cuda()
            for i, length in enumerate(lengths):
                length = length.item()
                if (mask[i].size(2) - length) > 0:
                    mask[i].narrow(2, length, mask[i].size(2) - length).fill_(1)
            x = x.masked_fill(mask, 0)
        return x, lengths


class InferenceBatchSoftmax(nn.Module):
    def forward(self, input_):
        if not self.training:
            return F.softmax(input_, dim=-1)
        else:
            return input_


class BatchRNN(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        rnn_type=nn.LSTM,
        bidirectional=False,
        batch_norm=True,
    ):
        super(BatchRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.batch_norm = (
            SequenceWise(nn.BatchNorm1d(input_size)) if batch_norm else None
        )
        self.rnn = rnn_type(
            input_size=input_size,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            bias=True,
        )
        self.num_directions = 2 if bidirectional else 1

    def flatten_parameters(self):
        self.rnn.flatten_parameters()

    def forward(self, x, output_lengths):
        if self.batch_norm is not None:
            x = self.batch_norm(x)

        #  x = nn.utils.rnn.pack_padded_sequence(x, output_lengths)
        x, h = self.rnn(x)
        # print(x)
        # print(x.size())
        # x, _ = nn.utils.rnn.pad_packed_sequence(x)
        if self.bidirectional:
            x = (
                x.view(x.size(0), x.size(1), 2, -1)
                .sum(2)
                .view(x.size(0), x.size(1), -1)
            )  # (TxNxH*2) -> (TxNxH) by sum
        return x


class Lookahead(nn.Module):
    # Wang et al 2016 - Lookahead Convolution Layer for Unidirectional Recurrent Neural Networks
    # input shape - sequence, batch, feature - TxNxH
    # output shape - same as input
    def __init__(self, n_features, context):
        super(Lookahead, self).__init__()
        assert context > 0
        self.context = context
        self.n_features = n_features
        self.pad = (0, self.context - 1)
        self.conv = nn.Conv1d(
            self.n_features,
            self.n_features,
            kernel_size=self.context,
            stride=1,
            groups=self.n_features,
            padding=0,
            bias=False,
        )

    def forward(self, x):
        x = x.transpose(0, 1).transpose(1, 2)
        x = F.pad(x, pad=self.pad, value=0)
        x = self.conv(x)
        x = x.transpose(1, 2).transpose(0, 1).contiguous()
        return x

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "n_features="
            + str(self.n_features)
            + ", context="
            + str(self.context)
            + ")"
        )


class DeepSpeech(pl.LightningModule):
    def __init__(
        self,
        dataD_cfg: DTWDataConfig,
        model_cfg: Union[UniDirectionalConfig, BiDirectionalConfig],
        precision: int,
        optim_cfg: Union[AdamConfig, SGDConfig],
        spect_cfg: SpectConfig,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model_cfg = model_cfg
        self.precision = precision
        self.optim_cfg = optim_cfg
        self.spect_cfg = spect_cfg
        self.data_cfg = dataD_cfg
        self.bidirectional = (
            True if OmegaConf.get_type(model_cfg) is BiDirectionalConfig else False
        )

        self.conv = MaskConv(
            nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
                nn.BatchNorm2d(32),
                nn.Hardtanh(0, 20, inplace=True),
                nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
                nn.BatchNorm2d(32),
                nn.Hardtanh(0, 20, inplace=True),
            )
        )
        # comme on a des batch de 1 , faut enlever le mask
        # batch > 1 faut changer la collate_fn

        # Based on above convolutions and spectrogram size using conv formula (W - F + 2P)/ S+1
        rnn_input_size = int(math.floor((16000 * 0.02) / 2) + 1)
        rnn_input_size = int(math.floor(rnn_input_size + 2 * 20 - 41) / 2 + 1)
        rnn_input_size = int(math.floor(rnn_input_size + 2 * 10 - 21) / 2 + 1)
        rnn_input_size *= 32

        self.rnns = nn.Sequential(
            BatchRNN(
                input_size=rnn_input_size,
                hidden_size=1024,
                rnn_type=nn.LSTM,
                bidirectional=False,
                batch_norm=False,
            ),
            *(
                BatchRNN(
                    input_size=1024,
                    hidden_size=1024,
                    rnn_type=nn.LSTM,
                    bidirectional=False,
                )
                for x in range(5 - 1)
            )
        )

        self.lookahead = nn.Sequential(
            # consider adding batch norm?
            Lookahead(1024, context=20),
            nn.Hardtanh(0, 20, inplace=True),
        )

        # changer la metric et la loss

        self.inference_softmax = InferenceBatchSoftmax()

        if self.data_cfg.labels == "with":
            print("loss with labels")
            self.criterion = DTWLosslabels(representation=self.data_cfg.representation)

        elif self.data_cfg.labels == "without":
            print("loss without labels")
            self.criterion = DTWLosswithoutlabels(
                representation=self.data_cfg.representation
            )

    def forward_once(self, x):
        lengths = torch.tensor(x.size(1))
        lengths = lengths.cpu().int()
        output_lengths = self.get_seq_lens(lengths)
        output_lengths.unsqueeze_(0)
        x = x.unsqueeze(0)
        x, _ = self.conv(x, output_lengths)

        sizes = x.size()
        x = x.view(
            sizes[0], sizes[1] * sizes[2], sizes[3]
        )  # Collapse feature dimension
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # TxNxH

        for rnn in self.rnns:
            x = rnn(x, output_lengths)

        if not self.bidirectional:  # no need for lookahead layer in bidirectional
            x = self.lookahead(x)

        #x = self.fc(x)
        #x = x.transpose(0, 1)
        # identity in training mode, softmax in eval mode
        #x = self.inference_softmax(x)

        x = torch.squeeze(x)

        #print(x.size())

        if self.data_cfg.representation == "gauss":
            x = gaussrep(x)
            x = torch.squeeze(x)

        return x, output_lengths

    def forward(self, input1, input2, input3):
        # forward pass of input 1
        output1, _ = self.forward_once(input1)
        # forward pass of input 2
        output2, _ = self.forward_once(input2)
        # forward pass of input 2
        output3, _ = self.forward_once(input3)

        return output1, output2, output3

    def training_step(self, batch, batch_idx):
        data = batch
        TGT, OTH, X = data[0], data[1], data[2]
        id_triplets = data[3]
        labels = data[4]

        output1, output2, output3 = self.forward(TGT, OTH, X)
        # out = out.transpose(0, 1)  # TxNxH
        # out = out.log_softmax(-1)

        loss = self.criterion(output1, output2, output3, labels)
        self.log('loss_train', loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data = batch
        # print(batch)

        TGT, OTH, X = data[0], data[1], data[2]
        id_triplets = data[3]
        labels = data[4]

        output1, output2, output3 = self.forward(TGT, OTH, X)

        val_loss = self.criterion(output1, output2, output3, labels)
        self.log("val_loss", val_loss, prog_bar=True, on_epoch=True)

        # get real dtw
        TGT, OTH, X = TGT.numpy(), OTH.numpy(), X.numpy()
        tgt_X = compute_dtw(TGT,X, dist_for_cdist='cosine', norm_div=True)
        oth_X = compute_dtw(OTH,X, dist_for_cdist='cosine', norm_div=True)
        self.log('real_val_value', (oth_X - tgt_X) - labels.numpy(), prog_bar = True, on_epoch = True)



    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)




    def get_seq_lens(self, input_length):
        """
        Given a 1D Tensor or Variable containing integer sequence lengths, return a 1D tensor or variable
        containing the size sequences that will be output by the network.
        :param input_length: 1D Tensor
        :return: 1D Tensor scaled by model
        """
        seq_len = input_length
        for m in self.conv.modules():
            if type(m) == nn.modules.conv.Conv2d:
                seq_len = (
                    seq_len
                    + 2 * m.padding[1]
                    - m.dilation[1] * (m.kernel_size[1] - 1)
                    - 1
                ) // m.stride[1] + 1
        return seq_len.int()
