import tensorflow as tf
from tensorflow.contrib import rnn
from data_helper import *

lr = 1e-3 # 基础学习率
decay_rate = 0.96 # 学习率衰减率
decay_steps = 100   # 每多少步进行一次衰减
seq_len = 25 # 输入的句子长度
Embedding_Size = 300 # LSTM隐藏层节点数
nhidden = Embedding_Size
Batch_Size = 100
Train_Step = 1000
max_accuracy=0  # 统计最大精度

class LSTM_Model:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.x = tf.placeholder(shape=[Batch_Size, seq_len], dtype=tf.int32)
        self.y = tf.placeholder(shape=[Batch_Size, seq_len], dtype=tf.int32)
        self.length = tf.placeholder(shape=[Batch_Size], dtype=tf.int32)
        self.mask = tf.placeholder(shape=[Batch_Size, seq_len], dtype=tf.float32)
        self.keep_prob = tf.placeholder(dtype=tf.float32)

        ######### 全连接层 权重矩阵和偏置
        self.w = tf.Variable(tf.truncated_normal([nhidden, self.vocab_size]), dtype=tf.float32)
        self.b = tf.Variable(tf.zeros([self.vocab_size]))

        ########### 生成变量进行词嵌入
        self.embedding_mat = tf.Variable(tf.random_uniform([self.vocab_size, Embedding_Size], -1, 1))
        self.inputs = tf.nn.embedding_lookup(self.embedding_mat, self.x)
        self.dinputs = tf.nn.dropout(self.inputs, keep_prob=self.keep_prob)

        ############ lstm
        self.lstm_cell = rnn.BasicLSTMCell(nhidden, forget_bias=1.0, state_is_tuple=True)
        self.lstm = rnn.DropoutWrapper(self.lstm_cell, output_keep_prob=self.keep_prob)
        self.initial_state = self.lstm.zero_state(Batch_Size, dtype=tf.float32)
        ############ 每个step的状态都要用到 因此直接用了lstm的output
        self.output, _ = tf.nn.dynamic_rnn(self.lstm, self.dinputs, initial_state=self.initial_state, dtype=tf.float32,time_major=False)

        self.lstm_out = tf.reshape(self.output, [Batch_Size*seq_len, nhidden])

        ########## 全连接层
        self.prob = tf.reshape(tf.matmul(self.lstm_out, self.w) + self.b,[Batch_Size, seq_len, self.vocab_size])

        self.loss = tf.reduce_mean(tf.contrib.seq2seq.sequence_loss(self.prob,
                                                                    self.y,
                                                                    self.mask,
                                                                    average_across_timesteps=False,
                                                                    average_across_batch=True))
        self.cost = tf.reduce_sum(self.loss)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), 5) # 限制梯度的变化范围

        ########### 设置学习率衰减
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.learning_rate = tf.train.exponential_decay(lr, self.global_step, decay_steps, decay_rate, staircase=True)

        self.train = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(grads, tvars), global_step=self.global_step)

    def test(self, sess,Id2word,test_data):
        global max_accuracy
        x, l = test_data.get_test_data()
        Answer = test_data.Answer
        accuracy = 0
        predictions = []
        output = sess.run(self.prob, feed_dict={self.x: x,self.length: l, self.keep_prob: 1.0})
        for i in range(len(x)):
            prediction = np.argmax(output[i, l[i]-1, :])
            prediction = Id2word[prediction]
            predictions.append(prediction)
            # print(prediction, Answer[i])
            if prediction == Answer[i]:
                print(prediction)
                accuracy += 1

        if accuracy > max_accuracy:
            max_accuracy = accuracy
            with open('./test/rnn_prediction.txt', 'w', encoding='utf-8') as f:
                f.write('\n'.join(predictions))
        return float(accuracy) / float(len(x))


if __name__ == '__main__':
    train_data = build_trainData(seq_len + 1)
    test_data = build_testData(seq_len,train_data.word2Id,train_data.Id2word)

    model = LSTM_Model(len(train_data.word2Id))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(Train_Step + 1):
            input_batch, output_batch, length_batch, mask_batch = train_data.next_batch(Batch_Size)
            _, _loss, _step = sess.run([model.train, model.loss, model.global_step],
                                      feed_dict={model.x: input_batch, model.y: output_batch,
                                                 model.length: length_batch, model.mask: mask_batch,
                                                 model.keep_prob: 1.0})

            if i % 30 == 0:
                print('当前步数为:{}'.format(_step))
                print('Loss为:{}'.format( _loss))
                print("当前学习率为：{}".format(sess.run(model.learning_rate)))
                print('测试集上的正确率为: {}'.format(model.test(sess,train_data.Id2word,test_data)))
                print("---------------------")

