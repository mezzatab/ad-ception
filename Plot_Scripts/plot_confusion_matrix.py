from keras.layers import Input, Dense
from keras.models import Model
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.layers import LSTM
from keras.layers import GlobalMaxPool1D
from keras.layers import Dropout
from keras.models import Sequential 

def LSTM_Classifier(embDim=128, lstmDim=60, hidDim=50, outDim=6, maxlen=50, max_features=20000):
    model=Sequential()
    model.add(Embedding(max_features, embDim, input_length=maxlen))
    model.add(LSTM(lstmDim, return_sequences=True, name='lstm_layer'))
    model.add(GlobalMaxPool1D())
    model.add(Dropout(0.1))
    model.add(Dense(hidDim, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(outDim, activation='sigmoid'))
#    model=Model(inp,x)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
#    model.summary()
    return model 