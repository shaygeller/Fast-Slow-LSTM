from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.init
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
# from torch.nn.init import xavier
import LNLSTM
import FSRNN
import helper

import reader
import config

import time
import numpy as np

from NEW_LSTM import LayerNormLSTM, LSTM

vocab_size = 50
weight = torch.ones(vocab_size)
# weight[0] = 0.0
criterion = nn.CrossEntropyLoss(weight=weight, reduction="none").cuda()
# criterion = nn.CrossEntropyLoss(ignore_index=0)
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
args = config.get_config()


class PTB_Model(nn.Module):
    def __init__(self, embedding_dim=args.embed_size, num_steps=args.num_steps, batch_size=args.batch_size,
                  vocab_size=args.vocab_size, num_layers=args.num_layers, dp_keep_prob=args.keep_prob,name=None):
        super(PTB_Model, self).__init__()
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.vocab_size = vocab_size

        self.n_layers = 1
        self.n_hidden = args.cell_size

        self.F_size = args.cell_size
        self.S_size = args.hyper_size
        self.dp_keep_prob = dp_keep_prob
        self.num_steps = num_steps
        self.emb_size = embedding_dim
        self.is_train = False

        self.embedding = nn.Embedding(self.vocab_size, self.emb_size)

        # self.F_cells = [LNLSTM.LN_LSTMCell(self.F_size, use_zoneout=True, is_training=self.is_train,
        #                                    zoneout_keep_h=args.zoneout_h, zoneout_keep_c=args.zoneout_c)
        #                 for _ in range(args.fast_layers)]
        #
        # self.S_cell  = LNLSTM.LN_LSTMCell(self.S_size, use_zoneout=True, is_training=self.is_train,
        #                                   zoneout_keep_h=args.zoneout_h, zoneout_keep_c=args.zoneout_c)

        # self.F_cell1 = nn.LSTM(self.emb_size, self.F_size, self.n_layers,
        #                     dropout=0, batch_first=True,bias=False)
        # self.S_cell = nn.LSTM(self.F_size, self.S_size, self.n_layers,
        #                     dropout=0, batch_first=True, bias=False)
        # self.F_cell2 = nn.LSTM(self.S_size, self.F_size, self.n_layers,
        #                     dropout=0, batch_first=True, bias=False)
        #
        self.F_cell1 = LSTM(self.emb_size, self.F_size, bias=True, dropout=0.0,
                      dropout_method='pytorch')

        self.S_cell = LSTM(self.F_size, self.S_size, bias=True, dropout=0.0,
                      dropout_method='pytorch')

        self.F_cell2 = LSTM(self.S_size, self.F_size, bias=True, dropout=0.0,
                      dropout_method='pytorch')


        self.dropout0 = nn.Dropout(self.dp_keep_prob)
        self.dropout1 = nn.Dropout(self.dp_keep_prob)
        self.dropout2 = nn.Dropout(self.dp_keep_prob)
        self.dropout3 = nn.Dropout(self.dp_keep_prob)

        self.fc = nn.Linear(self.F_size, vocab_size,bias=True)
        # for p in self.fc.parameters():
        #     if p.ndimension()>1:
        #         print(p.data)
        #         torch.nn.init.orthogonal_(p.data)
        #         print(p.data)
        # self.F10_norm = nn.LayerNorm(self.F_size, eps=1e-03)
        # self.F11_norm = nn.LayerNorm(self.F_size, eps=1e-03)
        # self.F20_norm = nn.LayerNorm(self.F_size, eps=1e-03)
        # self.F21_norm = nn.LayerNorm(self.F_size, eps=1e-03)
        # self.S10_norm = nn.LayerNorm(self.S_size, eps=1e-03)
        # self.S11_norm = nn.LayerNorm(self.S_size, eps=1e-03)


        # self.FS_cell = FSRNN.FSRNNCell(self.F_cells, self.S_cell, args.keep_prob, self.is_train)
        # self._initial_state = self.FS_cell.zero_state(batch_size, torch.FloatTensor)

    def forward(self,inputs, hidden):

        ''' Forward pass through the network.
            These inputs are x, and the hidden/cell state `hidden`. '''
        zoneout = False

        F_state = hidden[0]
        S_state = hidden[1]

        inputs = self.embedding(inputs)
        inputs = self.dropout0(inputs)

        # Create input for LSTM with num_step=1
        # inputs = inputs.view(self.batch_size,1,self.emb_size)


        F_output, F_state_new = self.F_cell1(inputs, F_state)
        # F_output, F_state_new = self.F_cell1(inputs.view(inputs.shape[0],1,-1), F_state)
        # F_output = self.F1_norm(F_output)
        # F_state_new_0 = self.F10_norm(F_state_new[0])
        # F_state_new_1 = self.F11_norm(F_state_new[1])
        # F_state_new = (F_state_new_0, F_state_new_1)
        # F_output = F_state_new_0.view(model.batch_size,-1,model.F_size)
        if zoneout:
            new_h, new_c = helper.zoneout(F_state_new[0], F_state_new[1], F_state[0], F_state[1], 0.9,0.5, True)
            F_state_new = (new_h, new_c)
        F_output_drop = self.dropout1(F_output)


        S_output, S_state_new = self.S_cell(F_output_drop, S_state)
        # S_state_new_0 = self.S10_norm(S_state_new[0])
        # S_state_new_1 = self.S11_norm(S_state_new[1])
        # S_state_new = (S_state_new_0, S_state_new_1)
        # S_output = S_state_new_0.view(model.batch_size,-1,model.S_size)

        if zoneout:
            new_h, new_c = helper.zoneout(S_state_new[0], S_state_new[1], S_state[0], S_state[1], 0.9, 0.5, True)
            S_state_new = (new_h, new_c)
        S_output_drop = self.dropout2(S_output)

        F_output, F_state_new2 = self.F_cell2(S_output_drop, F_state_new)
        # F_state_new2_0 = self.F20_norm(F_state_new2[0])
        # F_state_new2_1 = self.F21_norm(F_state_new2[1])
        # F_state_new2 = (F_state_new2_0, F_state_new2_1)
        # F_output = F_state_new2_0.view(model.batch_size,-1,model.F_size)

        if zoneout:
            new_h, new_c = helper.zoneout(F_state_new[0], F_state_new[1], F_state[0], F_state[1], 0.9, 0.5, True)
            F_state_new2 = (new_h, new_c)
        F_output_drop = self.dropout3(F_output)


        # for i in range(2, 5):
        #     out, F_state = self.F_cell2(out, F_state)

        # # Stack up LSTM outputs using view
        # # you may need to use contiguous to reshape the output
        out_flatten = F_output_drop.contiguous().view(-1, self.n_hidden)

        # out = self.fc(out_flatten)

        hidden = (F_state_new2,S_state_new)

        return out_flatten,hidden
        # return out,hidden









        # # state = self._initial_state
        # # F_state =(torch.randn(1, 1, 3),
        # #   torch.randn(1, 1, 3))
        # # S_state
        # outputs = []
        # inputs = self.embedding(inputs)
        # for time_step in range(self.num_steps):
        #     step_input = inputs[:,time_step,:]
        #     # out , state = self.FS_cell(inputs[:,time_step,:],state)
        #
        #     step_input = self.dropout(step_input)
        #
        #     F_output, F_state = self.F_cell1(step_input, F_state)
        #
        #     F_output_drop = self.dropout(F_output)
        #
        #     S_output, S_state = self.S_cell(F_output_drop, S_state)
        #     S_output_drop = self.dropout(S_output)
        #
        #     F_output, F_state = self.F_cell2(S_output_drop, F_state)
        #
        #     # for i in range(2, self.fast_layers):
        #     #     print("LALA")
        #     #     F_output, F_state = self.fast_cells[i](F_output[:, 0:1] * 0.0, F_state)
        #
        #     F_output_drop = self.dropout(F_output)
        #
        #     outputs.append(F_output_drop)

        # output = torch.cat(outputs,dim =1).view([-1,self.F_size]).cuda()
        #
        # softmax_w = helper.orthogonal_initializer([self.F_size, self.vocab_size]).cuda()
        # softmax_b = helper.orthogonal_initializer([self.vocab_size]).cuda()
        #
        # logits = torch.mm(output , softmax_w) + softmax_b
        #
        # return logits.view([self.num_steps,self.batch_size,self.vocab_size])

    def init_hidden(self, batch_size):
        # ''' Initializes hidden state '''
        # # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # # initialized to zero, for hidden state and cell state of LSTM
        hidden_stats = []
        number_of_latms = 2
        weight = next(self.parameters()).data

        train_on_gpu = True
        if (train_on_gpu):
            F_hidden = (weight.new(self.n_layers, batch_size, self.F_size).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.F_size).zero_().cuda())
        else:
            F_hidden = (weight.new(self.n_layers, batch_size, self.F_size).zero_(),
                      weight.new(self.n_layers, batch_size, self.F_size).zero_())


        if (train_on_gpu):
            S_hidden = (weight.new(self.n_layers, batch_size, self.S_size).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.S_size).zero_().cuda())
        else:
            S_hidden = (weight.new(self.n_layers, batch_size, self.S_size).zero_(),
                      weight.new(self.n_layers, batch_size, self.S_size).zero_())

        return (F_hidden,S_hidden)



def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def run_epoch(model, data, is_train=False, lr=1.0):
    """Runs the model on the given data."""
    if is_train:
        model.is_train = True
        model.train()
    else:
        model.eval()
    
    epoch_size = ((len(data) // model.batch_size) - 1) // model.num_steps
    start_time = time.time()
    costs = 0.0
    iters = 0.0

    h = model.init_hidden(model.batch_size)
    print("Before")
    for p in model.parameters():
        print(p.data.shape)
        print(p.data)
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.0)
    for step, (x, y) in enumerate(reader.ptb_iterator(data, model.batch_size, model.num_steps)):

        inputs = Variable(torch.from_numpy(x.astype(np.int64)).transpose(0, 1).contiguous()).cuda()
        inputs = torch.transpose(inputs, 0, 1)

        # h = model.init_hidden(model.batch_size)

        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        h_new=[]
        for tup in h:
            # h_temp = tuple([each.data for each in tup])
            h_temp = repackage_hidden(tup)
            h_new.append(h_temp)

        model.zero_grad()
        outputs = []
        # print(inputs.data)

        for time_step in range(model.num_steps):
            step_input = inputs[:,time_step]
            output, h = model(step_input, h_new)
            outputs.append(output)

        outputs = torch.cat(outputs,dim =1).view([-1,model.F_size]).cuda()
        outputs = model.fc(outputs)

        # outputs, h = model(inputs, h_new)
        # TODO: Check initial weights in Tensorflow and Pytorch implementations - Need to init Bias to zeroes
        # TODO: Classify only steps first, then chech if can extend to full batch matrix for better speed
        targets = Variable(torch.from_numpy(y.astype(np.int64)).transpose(0, 1).contiguous()).cuda()
        tt = torch.squeeze(targets.view(-1, model.batch_size * model.num_steps))

        loss = criterion(outputs, tt).mean()
        loss.backward()
        # loss = criterion(outputs.view(-1, model.vocab_size), tt)
        with torch.no_grad():

            print("Loss ", loss.item()* model.num_steps)
            # costs += loss.data[0] * model.num_steps
            costs += loss.item() * model.num_steps
            # costs += loss.item()
            iters += model.num_steps

        if is_train:
            # print("Before")
            # for p in model.parameters():
            #     print(p.grad)

            # loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            # print("After")
            # for p in model.parameters():
            #     print(p.shape)
            #     print(p.grad)

            # for p in model.parameters():
            #     p.data.add_(-lr, p.grad.data)
            optimizer.step()

            if step % (epoch_size // 10) == 10:
                print("Loss ", loss.item()* model.num_steps)
                print("{} perplexity: {:8.2f} speed: {} wps".format(step * 1.0 / epoch_size, costs / (iters* 0.69314718056),
                                      iters * model.batch_size / (time.time() - start_time)))
    return costs / (iters* 0.69314718056)

if __name__ == "__main__":

    raw_data = reader.ptb_raw_data(data_path=args.data_path)
    train_data, valid_data, test_data, word_to_id, id_to_word = raw_data
    vocab_size = len(word_to_id)
    print('Vocabulary size: {}'.format(vocab_size))
    model = PTB_Model(embedding_dim=args.embed_size, num_steps=args.num_steps, batch_size=args.batch_size,
                      vocab_size=vocab_size, num_layers=args.num_layers, dp_keep_prob=args.keep_prob)
    model = model.cuda()

    # # Init weights
    with torch.no_grad():
        for p in model.parameters():
            if p.ndimension() > 1:
                print(p.data.shape)
                print(p.data)
                torch.nn.init.orthogonal_(p.data)
                print(p.data)
            elif p.ndimension() == 1:
                print(p.data.shape)
                torch.nn.init.constant_(p.data,0.0)

    # # Init weights
    # with torch.no_grad():
    #     for p in model.parameters():
    #         p.uniform_(-0.01, 0.01)


    lr = args.lr_start
    # decay factor for learning rate
    lr_decay_base = args.lr_decay_rate
    # we will not touch lr for the first m_flat_lr epochs
    m_flat_lr = 14.0

    print("########## Training ##########################")

    for epoch in range(args.max_max_epoch):
        lr_decay = lr_decay_base ** max(epoch - m_flat_lr, 0)
        lr = lr * lr_decay # decay lr if it is time
        # lr = 0.1
        train_p = run_epoch(model, train_data, True, lr)
        print('Train perplexity at epoch {}: {:8.2f}'.format(epoch, train_p))
        print('Validation perplexity at epoch {}: {:8.2f}'.format(epoch, run_epoch(model, valid_data)))


    print("########## Testing ##########################")
    model.batch_size = 1 # to make sure we process all the data
    print('Test Perplexity: {:8.2f}'.format(run_epoch(model, test_data)))
    with open(args.save, 'wb') as f:
        torch.save(model, f)

