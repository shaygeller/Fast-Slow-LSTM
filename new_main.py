from __future__ import print_function

import os
from keras.layers import Input, Embedding, LSTM, Dense, TimeDistributed
from keras.models import Model
from keras.utils import plot_model
from keras import backend as K


from keras import optimizers

import reader
import config

import time
import numpy as np

import tensorflow as tf
args = config.get_config()


def perplexity(y_true, y_pred):
    # cross_entropy = tf.reduce_mean(K.sparse_categorical_crossentropy(y_true, y_pred))
    cross_entropy = tf.reduce_sum(K.sparse_categorical_crossentropy(y_true, y_pred)) / 128
    # cross_entropy = K.mean(K.sparse_categorical_crossentropy(y_true, y_pred))
    # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_true,1), logits=y_pred)
    perplexity = K.exp(cross_entropy)

    # There is no log_2(x) (log with base 2) in Tensorflow, so we approximate it with log_2(x)=log_e(x)\log_e(2)
    # we can calc 1\log_e(2) which is 1.442695 and multiple.
    oneoverlog2 = 1.442695
    result = K.log(perplexity) * oneoverlog2

    return cross_entropy / (5 * 0.69314718056)

    # To check if we calculate perplexity right, return perplexity and switch the activation function of the
    # Dense layer inside the TimeDistributedLayer to softmax. You will need to get first perplexity
    # in size of the vocabulary
    return result


def get_keras_model(embedding_dim, num_steps, batch_size, vocab_size, num_layers=3, f1_size=700, s1_size=400, f2_size=700):


    # Headline input: meant to receive sequences of 100 integers, between 1 and 10000.
    # Note that we can name any layer by passing it a "name" argument.
    # main_input = Input(shape=(100,), dtype='int32', name='main_input')
    inputs1 = Input(shape=(None,))

    # This embedding layer will encode the input sequence
    # into a sequence of dense 512-dimensional vectors.
    x = Embedding(output_dim=embedding_dim, input_dim=vocab_size, input_length=num_steps)(inputs1)

    # Fast1
    fast1_out, fast1_out_again, fast1_state = LSTM(f1_size, return_state=True,
                                                   return_sequences=True, use_bias=False)(x)

    slow1_out, slow1_out_again, slow1_state = LSTM(s1_size, return_state=True,
                                                   return_sequences=True, use_bias=False)(fast1_out)

    fast2_out = LSTM(f2_size, return_state=False,
                     return_sequences=True, use_bias=False)([slow1_out, fast1_out_again, fast1_state])

    for i in range(2, num_layers):
            fast2_out = LSTM(f2_size, return_state=False, return_sequences=True, use_bias=False)(fast2_out)

    out = TimeDistributed(Dense(vocab_size, activation='linear'))(fast2_out)
    # out = TimeDistributed(Dense(vocab_size, activation='softmax'))(fast2_out)

    model = Model(inputs=inputs1, outputs=out)


    # summarize layers
    print(model.summary())
    # plot graph
    # plot_model(model, to_file='Fast_slow_rnn.png')

    return model


if __name__ == "__main__":

    raw_data = reader.ptb_raw_data(data_path=args.data_path)
    train_data, valid_data, test_data, word_to_id, id_to_word = raw_data
    vocab_size = len(word_to_id)
    print('Vocabulary size: {}'.format(vocab_size))
    # model = PTB_Model(embedding_dim=args.embed_size, num_steps=args.num_steps, batch_size=args.batch_size,
    #                   vocab_size=vocab_size, num_layers=args.num_layers, dp_keep_prob=args.keep_prob)

    model = get_keras_model(embedding_dim=args.embed_size, num_steps=args.num_steps, batch_size=args.batch_size,
                            vocab_size=vocab_size, num_layers=args.num_layers)


    lr = args.lr_start
    # decay factor for learning rate
    lr_decay_base = args.lr_decay_rate
    # we will not touch lr for the first m_flat_lr epochs

    print("########## Training ##########################")

    optimizer = optimizers.Adam(lr=lr, decay=lr_decay_base)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=[perplexity])

    model.fit_generator(reader.ptb_iterator(train_data, args.batch_size, args.num_steps),
                        steps_per_epoch=len(train_data)//(args.batch_size*args.num_steps),
                        epochs=args.max_epoch,
                        verbose=1,
                        validation_data=reader.ptb_iterator(valid_data, args.batch_size, args.num_steps),
                        validation_steps=len(valid_data)//(args.batch_size*args.num_steps),)
        # print('Train perplexity at epoch {}: {:8.2f}'.format(epoch, train_p))
        # print('Validation perplexity at epoch {}: {:8.2f}'.format(epoch, run_epoch(model, valid_data)))
    timestr = time.strftime("%Y%m%d-%H%M%S")

    model.save(timestr+'my_model.h5')  # creates a HDF5 file 'my_model.h5'

    #
    #
    # print("########## Testing ##########################")
    # model.batch_size = 1 # to make sure we process all the data
    # print('Test Perplexity: {:8.2f}'.format(run_epoch(model, test_data)))
    # with open(args.save, 'wb') as f:
    #     torch.save(model, f)

