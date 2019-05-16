from model import get_model, decode
from data_utils import get_data, seq2seq_data_generator


batch_size = 64
epochs = 3

train_data, test_data = get_data()

gen = seq2seq_data_generator(train_data, batch_size)

model = get_model()
model.load_weights('weights1.h5')

model.fit_generator(gen, len(train_data[0]), epochs=epochs)

model.save_weights('weights.h5')
