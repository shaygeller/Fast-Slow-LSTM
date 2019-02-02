from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable

class FSRNNCell(object):
    
    def __init__(self, fast_cells,slow_cell, keep_prob = 1.0, training = True):
        super(FSRNNCell, self).__init__()
        self.fast_layers = len(fast_cells)
        assert self.fast_layers >=2 , 'Atleast 2 fast layers are needed.'

        self.fast_cells = fast_cells
        self.slow_cell = slow_cell
        self.keep_prob = keep_prob
        if not training:
            self.keep_prob = 1.0

        self.dropout = nn.Dropout(p = 1-self.keep_prob)

    def __call__(self,inputs,state):
            F_state = state[0]
            S_state = state[1]

            inputs = self.dropout(inputs)
            
            F_output, F_state = self.fast_cells[0](inputs, F_state)
            
            F_output_drop = self.dropout(F_output)

            S_output, S_state = self.slow_cell(F_output_drop, S_state)
            S_output_drop = self.dropout(S_output)

            F_output, F_state = self.fast_cells[1](S_output_drop, F_state)

            for i in range(2, self.fast_layers):
                F_output, F_state = self.fast_cells[i](F_output[:, 0:1] * 0.0, F_state)


            F_output_drop = self.dropout(F_output)

            return F_output_drop, (F_state, S_state)

    def zero_state(self, batch_size, dtype = torch.FloatTensor):
            F_state = self.fast_cells[0].zero_state(batch_size, dtype)
            S_state = self.slow_cell.zero_state(batch_size, dtype)
            

            return (F_state, S_state)
