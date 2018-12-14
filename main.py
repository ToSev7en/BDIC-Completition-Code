import os
import sys
import bz2
from bz2 import decompress
import codecs
import tqdm


import keras
from keras.models import Model
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Activation
from keras.layers import Input, Dense, Embedding, Conv1D, Conv2D, MaxPooling1D, MaxPool2D
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.layers import SpatialDropout1D, concatenate
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers.recurrent import LSTM,SimpleRNN
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from keras.utils.np_utils import to_categorical
from keras.models import load_model
from keras.layers import Lambda,BatchNormalization
from keras import backend as K

import jieba
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 导入自定义库
from utils.data_utils import (
    split_dataset,
    clean_str,
    build_vocab,
    get_tokens,
    seg_corpus
)

# 基本参数配置

BASE_DIR = '/Users/tsw/ScenicSpotReviews'

W2V_DIR = BASE_DIR + '/embeddings/'

TEXT_DATA_DIR = BASE_DIR + '/data/'

MAX_NUM_WORDS = 33950

MAX_SEQUENCE_LENGTH = 150 # 每篇文章选取150个词

MAX_NB_WORDS = 80000 # 将字典设置为含有1万个词84480

EMBEDDING_DIM = 300 # 词向量维度，300维

VALIDATION_SPLIT = 0.1 # 测试集大小，全部数据的10%

BATCH_SIZE = 256

EPOCHS = 15

NUM_LABELS = 3


# 加载训练集
print('Loading Training Set')
df_train_dataset = pd.read_csv('./data/training-inspur.csv', encoding='utf-8')

# 加载测试集
print('Loading Test Set')
df_test_dataset = pd.read_csv('./data/Preliminary-texting.csv', encoding='utf-8')

# 提取数据集中所需的字段
df_train_dataset = df_train_dataset[['COMMCONTENT', 'COMMLEVEL']]
df_test_dataset = df_test_dataset[['COMMCONTENT']]

# 合并数据集用于构建词汇表
df_all_dataset = pd.concat([df_train_dataset, df_test_dataset], ignore_index=True)

# 对所有文本分词
print('texts segmentation...')
seged_text = seg_corpus(df_all_dataset['COMMCONTENT'])

# 将分词后的数据并入 df_all_dataset
df_all_dataset['COMMCONTENT_SEG'] = pd.DataFrame(seged_text,columns=['COMMCONTENT_SEG'])

# text_corpus
text_corpus = df_all_dataset['COMMCONTENT_SEG']
# 传入我们词向量的字典
tokenizer = Tokenizer(num_words=MAX_NB_WORDS) 
# 传入我们的训练数据，得到训练数据中出现的词的字典
tokenizer.fit_on_texts(text_corpus) 

# 根据训练数据中出现的词的字典，将训练数据转换为sequences
dataset_sequences = tokenizer.texts_to_sequences(text_corpus) 

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

padded_dataset_sequences = pad_sequences(dataset_sequences, maxlen=MAX_SEQUENCE_LENGTH)
print('padded_dataset_sequences.shape:',padded_dataset_sequences.shape)


df_test_dataset_seg = df_all_dataset['COMMCONTENT_SEG'][20000:]
test_dataset_sequences = tokenizer.texts_to_sequences(df_test_dataset_seg)
padded_test_dataset_sequences = pad_sequences(test_dataset_sequences, maxlen=MAX_SEQUENCE_LENGTH)

# 划分训练集和测试集
print('split train_X,valid_X,train_y,valid_y')
train_X,valid_X,train_y,valid_y =train_test_split(padded_dataset_sequences[:df_train_dataset.shape[0]], 
                                                  df_all_dataset['COMMLEVEL'][:df_train_dataset.shape[0]], 
                                                  test_size=0.1)


# label one-hot 表示
labels = df_all_dataset['COMMLEVEL'].dropna().map(int)#.values.tolist()
labels = to_categorical(labels-1) 

vocab,vocab_freqs = build_vocab(df_all_dataset['COMMCONTENT_SEG'])
vocab_size = min(MAX_NB_WORDS, len(vocab_freqs)) + 2
word2index = {x[0]: i+2 for i, x in enumerate(vocab_freqs.most_common(MAX_NB_WORDS))}
word2index["PAD"] = 0
word2index["UNK"] = 1
index2word = {v:k for k, v in word2index.items()}
len(word2index)

# 解压 bz2 的词向量压缩文件
if not os.path.exists('./embeddings/sgns.weibo.word'):

    print('Start unbz2 embeddings file')

    bz2file_path = "./embeddings/sgns.weibo.word.bz2"

    un_path="./embeddings"

    filename = 'sgns.weibo'

    newfilepath = os.path.join(un_path, filename + '.word')

    with open(newfilepath, 'wb') as new_file, bz2.BZ2File(bz2file_path, 'rb') as file:
        for data in tqdm.tqdm(iter(lambda : file.read(100 * 1024), b'')):
            new_file.write(data)

    print('Unbz2 embeddings file done!')


print('Indexing word embeddings.')  
embeddings_index = {}
with codecs.open('./embeddings/sgns.weibo.word','r','utf-8') as f:
    f = f.readlines()
    for i in f[1:]:
        values = i.strip().split(' ')
        word = str(values[0])
        embedding = np.asarray(values[1:],dtype='float')
        embeddings_index[word] = embedding
print('word embedding',len(embeddings_index))


nb_words = min(MAX_NB_WORDS,len(word2index))

word_embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
for word, i in word2index.items():
    if i > MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(str(word).upper())
    if embedding_vector is not None:
        word_embedding_matrix[i] = embedding_vector


def get_bigru_cnn_model():
    inp = Input(shape=(MAX_SEQUENCE_LENGTH,))
    x =  Embedding(input_dim = nb_words+1, 
                             output_dim = EMBEDDING_DIM, 
                             weights=[word_embedding_matrix], 
                             input_length=MAX_SEQUENCE_LENGTH, 
                             mask_zero=False,
                             trainable=True
                            )(inp)
    x = SpatialDropout1D(0.3)(x)
    x = Bidirectional(GRU(80, return_sequences=True))(x)
    
    x1 = Conv1D(32, kernel_size=2, padding="valid", kernel_initializer="he_uniform")(x)
    avg_pool = GlobalAveragePooling1D()(x1)
    # max_pool = GlobalMaxPooling1D()(x1)
    kmax_pool = Lambda(lambda x: K.max(x, axis=1), output_shape=(32,))(x1)
    conc1 = concatenate([avg_pool, kmax_pool])
    
    x2 = Conv1D(32, kernel_size=3, padding="valid", kernel_initializer="he_uniform")(x)
    avg_pool2 = GlobalAveragePooling1D()(x2)
    # max_pool2 = GlobalMaxPooling1D()(x2)
    kmax_pool2 = Lambda(lambda x: K.max(x, axis=1), output_shape=(32,))(x2)
    conc2 = concatenate([avg_pool2, kmax_pool2])
    
    
    merge = concatenate([conc1, conc2])
    
    drop_merge = Dropout(0.25)(merge)
    
    outp = Dense(3, activation="softmax")(drop_merge)

    model = Model(inputs=inp, outputs=outp)
    
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    
    return model

print('Training model...')
def train(**Kwargs):
    """
    训练
    """
    rnn_cnn_model = get_bigru_cnn_model()
    filepath = "./checkpoints/weights-improvement-{epoch:02d}-{val_acc:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,mode='max')

    history = rnn_cnn_model.fit(x=train_X, y=to_categorical(train_y-1, num_classes=3), 
                        validation_data=(valid_X, to_categorical(valid_y-1, num_classes=3)[:]), 
                        batch_size=BATCH_SIZE, 
                        epochs=EPOCHS,
                        shuffle=True,
                        callbacks = [EarlyStopping(monitor='val_acc', patience=3, mode='auto'),checkpoint],
                        verbose=1)
    print("Training done.")


def predict(**Kwargs):
    """
    预测
    """
    best_model = load_model('./checkpoints/weights-improvement-04-0.6950.hdf5')

    y_pred_rnn_cnn = best_model.predict(padded_test_dataset_sequences, batch_size=1024)

    pooled_gru_conv_model_preds = np.argmax(y_pred_rnn_cnn,axis=1)[:]+1

    # pd.Series(pooled_gru_conv_model_preds).value_counts(normalize=True)

    np.savetxt("./submissions/weights-improvement-04-0.6950.txt", pooled_gru_conv_model_preds,fmt="%d")


def help():
    """
    打印帮助信息
    """
    print("help")


if __name__ == '__main__':

    if sys.argv[1] == 'train':
        train()
    if sys.argv[1] == 'predict':
        predict()