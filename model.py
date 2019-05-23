from keras.layers import *
from keras.models import Model
from data_utils import vectorize_en_word, mal_idx2token, en_tokens, START_TOKEN, STOP_TOKEN, mal_token2idx, mal_tokens, mal_maxlen
from keras.utils import to_categorical
import numpy as np
import keras.backend as K


def _stack_rnns(rnns, input_dim, initial_state=False):
    inp = Input(shape=(None, input_dim))
    x = inp
    if initial_state:
        initial_state = [Input((rnns[0].units,)) for _ in range(len(rnns[0].states))]
    else:
        initial_state = None
    state = initial_state
    for i, rnn in enumerate(rnns):
        if i != len(rnns) - 1:
            assert rnn.return_sequences == True
        assert rnn.return_state == True
        out = rnn(x, initial_state=state)
        x = out[0]
        state = out[1:]
    outs = [x] + state
    if initial_state:
        inp = [inp] + initial_state
    return Model(inp, outs)


def get_model():
    input_dim = len(en_tokens) + 1
    output_dim = len(mal_tokens) + 1
    encoder_input = Input((None, input_dim))
    encoder_lstms = [
        LSTM(256, return_state=True, return_sequences=True),
        LSTM(256, return_state=True, return_sequences=False)
    ]

    encoder = _stack_rnns(encoder_lstms, input_dim)
    encoder_out, h, c = encoder(encoder_input)

    decoder_input = Input((None, output_dim))

    decoder_lstms = [
        LSTM(256, return_state=True, return_sequences=True),
        LSTM(256, return_state=True, return_sequences=True)
    ]

    decoder = _stack_rnns(decoder_lstms, output_dim, initial_state=True)

    decoder_output = decoder([decoder_input, h, c])

    dense = Dense(output_dim, activation='softmax')

    dense_output = dense(decoder_output[0])

    model = Model([encoder_input, decoder_input], dense_output)

    decoder_input = Input((None, output_dim))
    decoder_state_input = [Input(( decoder_lstms[0].units,)) for _ in range(2)]
    decoder_output = decoder([decoder_input] + decoder_state_input)
    dense_out = dense(decoder_output[0])
    decoder = Model([decoder_input] + decoder_state_input, [dense_out] + decoder_output[1:])

    model.encoder = encoder
    model.decoder = decoder

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model


def decode(model, input_text):
    inp = vectorize_en_word(input_text)
    inp = to_categorical(inp, len(en_tokens) + 1)
    inp = np.expand_dims(inp, 0)
    _, h, c = model.encoder.predict(inp)
    curr_token = START_TOKEN
    output = ''
    while True:
        if len(output) == mal_maxlen:
            break
        decoder_out, h, c = model.decoder.predict(
            [to_categorical([[[mal_token2idx(curr_token)]]], len(mal_tokens) + 1), h, c])
        curr_token = chr(mal_idx2token(np.argmax(decoder_out[0][0])))
        if curr_token == STOP_TOKEN:
            break
        output += curr_token
    return output
