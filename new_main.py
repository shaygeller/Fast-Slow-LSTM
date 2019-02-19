from __future__ import print_function
from keras.models import load_model

import os
from keras.layers import Input, Embedding, LSTM, Dense, TimeDistributed
from keras.models import Model
from keras.utils import plot_model
from keras import backend as K
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import pickle

from keras import optimizers

import reader
import config

import time
import numpy as np

import tensorflow as tf



args = config.get_config()
tokenizer = Tokenizer()
is_train = False # True- traning model for scrach. False-Load trained model

def generate_text(seed_text, num_step, number_of_new_chars, model, idx_to_chars, chars_to_idx):
    """
        Generate new characters from the language model.

    :param seed_text: String of seed sentence. It's minimum length should be num_step.
                        Best if it will be related to the trained text.
    :param num_step: Step size of the input for the trained model.
    :param number_of_new_chars: Number of new characters to generate.
    :param model: Trained LM model.
    :param idx_to_chars: Index to char dictionary
    :param chars_to_idx: Char to index dictionaty
    :return:
    """

    # Trim too long seed sentence for the model
    seed_text = seed_text[:num_step].lower()
    new_sen = ""

    seed_text = [chars_to_idx[char] for char in seed_text]
    print("Before loop")
    for char in seed_text:
        new_sen = new_sen + idx_to_chars[char]
    print("Started from this sentence:\n",new_sen,"\n")

    for i in range(0, number_of_new_chars):
        print("Predicting for ", seed_text)
        # Predict all the characters
        predicted = model.predict(seed_text, batch_size=128, verbose=1)

        # Take only the last character (the new one)
        predicted_max = np.argmax(predicted[0], axis=1)[4]

        # Append the new character to the sentence
        new_sen = new_sen + idx_to_chars[predicted_max]
        print(new_sen)
        print()

        # Create the new seed sentence with the new char for the next iteration
        seed_text_new = [chars_to_idx[char] for char in new_sen[i + 1:]]
        seed_text = seed_text_new



def bpc(y_true, y_pred, num_steps, batch_size):
    """
        Calculate BPC (not perplexity)

    :param y_true: NpArray of labels . Should be size(batch_size,vocab_size)
    :param y_pred: NpArray of logit predictions. Should be size(batch_size,num_steps,vocab_size)
    :param num_steps: Number of steps of the LSTM
    :param batch_size: Batch size
    :return:
    """

    # cross_entropy = tf.reduce_mean(K.sparse_categorical_crossentropy(y_true, y_pred))
    cross_entropy = tf.reduce_sum(K.sparse_categorical_crossentropy(y_true, y_pred)) / batch_size
    # cross_entropy = K.mean(K.sparse_categorical_crossentropy(y_true, y_pred))
    # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_true,1), logits=y_pred)
    return cross_entropy / (num_steps * 0.69314718056)


    # another (I think better) option to calculate BPC, but it isn't like done in the original FS-RNN papar
    perplexity = K.exp(cross_entropy)

    # There is no log_2(x) (log with base 2) in Tensorflow, so we approximate it with log_2(x)=log_e(x)\log_e(2)
    # we can calc 1\log_e(2) which is 1.442695 and multiple.
    oneoverlog2 = 1.442695
    result = K.log(perplexity) * oneoverlog2

    # To check if we calculate perplexity right, return perplexity and switch the activation function of the
    # Dense layer inside the TimeDistributedLayer to softmax. You will need to get first perplexity
    # in size of the vocabulary
    return result


def get_keras_model(embedding_dim, num_steps, vocab_size, num_layers=3, f1_size=700, s1_size=400, f2_size=700):
    """
        Create FS-RNN model

    :param embedding_dim:
    :param num_steps:
    :param vocab_size:
    :param num_layers:
    :param f1_size:
    :param s1_size:
    :param f2_size:
    :return: Return model
    """

    inputs1 = Input(shape=(None,))

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

    # Summarize layers
    print(model.summary())
    # plot graph
    # plot_model(model, to_file='Fast_slow_rnn.png')

    return model


def bpc_wrapper(batch_size, num_steps):

    def perplexity(y_pred,y_true):
        return bpc(y_pred, y_true, batch_size=batch_size, num_steps=num_steps)

    return perplexity


if __name__ == "__main__":

    raw_data = reader.ptb_raw_data(data_path=args.data_path)
    train_data, valid_data, test_data, word_to_id, id_to_word = raw_data

    # Save dict to file for other modules to read
    # word_to_id
    pickle_out1 = open("word_to_id.pickle", "wb")
    pickle.dump(word_to_id, pickle_out1)
    pickle_out1.close()
    # id_to_word
    pickle_out2 = open("id_to_word.pickle", "wb")
    pickle.dump(id_to_word, pickle_out2)
    pickle_out2.close()


    vocab_size = len(word_to_id)
    print('Vocabulary size: {}'.format(vocab_size))
    # model = PTB_Model(embedding_dim=args.embed_size, num_steps=args.num_steps, batch_size=args.batch_size,
    #                   vocab_size=vocab_size, num_layers=args.num_layers, dp_keep_prob=args.keep_prob)

    model = get_keras_model(embedding_dim=args.embed_size, num_steps=args.num_steps,
                            vocab_size=vocab_size, num_layers=args.num_layers)


    lr = args.lr_start
    # decay factor for learning rate
    lr_decay_base = args.lr_decay_rate
    # we will not touch lr for the first m_flat_lr epochs
    perplexity_wrapped = bpc_wrapper(args.batch_size, args.num_steps)

    if is_train:
        print("########## Training ##########################")
        optimizer = optimizers.Adam(lr=lr, decay=lr_decay_base)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=[perplexity_wrapped])

        model.fit_generator(reader.ptb_iterator(train_data, args.batch_size, args.num_steps),
                            # steps_per_epoch=100,
                            steps_per_epoch=len(train_data)//(args.batch_size*args.num_steps),
                            epochs=args.max_epoch,
                            verbose=1,
                            validation_data=reader.ptb_iterator(valid_data, args.batch_size, args.num_steps),
                            validation_steps=len(valid_data)//(args.batch_size*args.num_steps),)
        timestr = time.strftime("%Y%m%d-%H%M%S")

        model.save(timestr+'my_model.h5')  # creates a HDF5 file 'my_model.h5'
        # model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'

    else:
        # Load trained model
        model_path = '20190218-223155my_model.h5'
        model = load_model(model_path, custom_objects={'perplexity': perplexity_wrapped})
        print("summery")
        print(model.summary())


    # Genetating New Sentences
    seed = "The name America was first recorded in 1507. Christie's auction house says a two-dimensional globe created by Martin Waldseemuller was the earliest recorded use of the term. The name was also used (together with the related term Amerigen) in the Cosmographiae Introductio, apparently written by Matthias Ringmann, in reference to South America. It was applied to both North and South America by Gerardus Mercator in 1538. America derives from Americus, the Latin version of Italian explorer Amerigo Vespucci's first name. The feminine form America accorded with the feminine names of Asia, Africa, and Europa"
    generate_text(seed, args.num_steps, 50, model, idx_to_chars=id_to_word, chars_to_idx=word_to_id)
