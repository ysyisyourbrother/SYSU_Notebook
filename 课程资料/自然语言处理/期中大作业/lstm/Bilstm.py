import tensorflow as tf
import os
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from  dataHelper import HelperClass
from scipy.stats import pearsonr

import matplotlib.pyplot as plt

#hyperparameters
decay_steps = 50   # 每多少步进行一次衰减
decay_rate = 0.96   # 学习率的衰减率
lr=1.0          # 学习率
epoch=1500        # 数据集循环使用多少次
batch_size=35           #每一批输入的句子对数量
n_inputs=100              # 每个词语的维度
n_steps=60               #一个句子 最多可能有多少个词语
n_hidden_unis=128       #rnn隐藏节点数
n_classes=128             #最后用多少维度表示句子向量

keep_prob=1 # 全连接层的dropout概率
fc1_unit=25 # 全连接层的隐藏层数量


em = HelperClass('',n_steps,n_inputs)
em.convert_embedding_file()

#Define weights
#weights:input weights+output weights
#进入RNN的cell之前，要经过一层hidden layer
#cell计算完结果后再输出到output hidden layer
#下面就定义cell前后的两层hidden layer，包括weights和biases



def Bi_RNN(X,X_len,weights,biases,reuse=True):
    # 先将batch和step换位置
    # 然后reshape，将batch和step合并
    # 随后后按照step 进行划分 满足双向rnn的要求
    # X_in = tf.transpose(X, [1, 0, 2])
    # X_in = tf.reshape(X_in, [-1, n_inputs])
    # X_in = tf.split(X_in, n_steps, 0)

    #cell 定义前向lstm
    #包含多少个节点，forget_bias:初始的forget定义为1，也就是不忘记，state_is_tuple：
    fw_lstm_cell=tf.nn.rnn_cell.BasicLSTMCell(n_hidden_unis,forget_bias=1.0,state_is_tuple=True,reuse=reuse)
    bw_lstm_cell=tf.nn.rnn_cell.BasicLSTMCell(n_hidden_unis,forget_bias=1.0,state_is_tuple=True,reuse=reuse)

    # 初始state,全部为0，慢慢的累加记忆 默认也为0
    # _init_state=lstm_cell.zero_state(batch_size,dtype=tf.float32)
    #outputs是一个list，每步的运算都会保存起来，time_majortime的时间点是不是在维度为1的地方，我们的放在第二个维度，28steps
    # outputs,states=tf.nn.dynamic_rnn(lstm_cell,X_in,initial_state=_init_state,time_major=False)

    # outputs, _, _ = tf.nn.static_bidirectional_rnn(cell_fw=fw_lstm_cell, cell_bw=bw_lstm_cell, inputs=X_in,dtype=tf.float32,sequence_length=X_len)

    outputs, states = tf.nn.bidirectional_dynamic_rnn(fw_lstm_cell, bw_lstm_cell, X, time_major=False, dtype=tf.float32,sequence_length=X_len)

    #hidden layer for outputs and final results
    results=tf.matmul(tf.concat((states[0][1],  states[1][1]),1),weights['out'])+biases['out']
    return results

def RNN(X,X_len,weights,biases,reuse=True):
    # 注释output的全连接层矩阵的大小
    X = tf.reshape(X, [-1, n_inputs])
    X_in = tf.matmul(X, weights['in']) + biases['in']  # 对应每个隐藏层节点加上偏置
    # 再变换为3维矩阵,(128 batch,28 steps,128 hidden)
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_unis])
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_unis, forget_bias=1.0, state_is_tuple=True, reuse=reuse)

    # 初始state,全部为0，慢慢的累加记忆
    _init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
    # outputs是一个list，每步的运算都会保存起来，time_majortime的时间点是不是在维度为1的地方，我们的放在第二个维度，28steps
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=_init_state, time_major=False,sequence_length=X_len)

    # hidden layer for outputs and final results
    results = tf.matmul(states[1], weights['out']) + biases['out']
    # tf.nn.relu(results, name=None)
    return results

def cal_pearsonr(x,y):
    res_x = tf.nn.moments(x, axes=0)
    res_y = tf.nn.moments(y, axes=0)
    conv = tf.reduce_sum((x - res_x[0]) * (y - res_y[0])) / tf.cast(tf.shape(x)[0], tf.float32)
    sigma = tf.sqrt(res_x[1] * res_y[1])
    return conv / sigma


def run_model():
    weights = {
        # (28,128)
        'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_unis])),
        # (128,28)
        'out': tf.Variable(tf.random_normal([n_hidden_unis, n_classes]))
    }

    biases = {
        # (128,)
        'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_unis, ])),
        # (10,)
        'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
    }
    # tf Graph input
    x1 = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
    x1_len = tf.placeholder(tf.int32, [None])
    x2 = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
    x2_len = tf.placeholder(tf.int32, [None])
    y = tf.placeholder(tf.float32, [None])


    with tf.variable_scope('lstmoutput'):
        lstm1=RNN(x1,x1_len,weights,biases,reuse=False)# 第一个lstm网络的输出
        lstm2=RNN(x2,x2_len,weights,biases,reuse=True) # 第二个lstm网络的输出，复用权重变量

    with tf.name_scope("fc"):
        # 用全连接层拟合相似度
        fc_in1 = lstm1 - lstm2  # 将两个向量加和
        fc_in2 = lstm1 * lstm2  # 将两个向量乘
        fc_in = tf.concat((fc_in1, fc_in2), 1) # 将二者拼接

        w1 = tf.Variable(tf.compat.v1.truncated_normal([n_classes*2, fc1_unit], stddev=0.1), dtype=tf.float32, name='w1')
        b1 = tf.Variable(tf.zeros([fc1_unit]), dtype=tf.float32, name='b1')
        w2 = tf.Variable(tf.compat.v1.truncated_normal([fc1_unit, 1], stddev=0.1), dtype=tf.float32, name='w2')
        b2 = tf.Variable(tf.zeros([1]), dtype=tf.float32, name='b2')

        fc1 = tf.matmul(fc_in, w1) + b1 # b会加到每个上面
        fc1 = tf.nn.relu(fc1)
        fc1 = tf.nn.dropout(fc1, keep_prob)
        fc2 = tf.matmul(fc1, w2) + b2
        fc2 = tf.nn.sigmoid(fc2)*5

    with tf.name_scope('consinLoss'):
        # 定义余弦损失函数
        s1 = tf.sqrt(tf.reduce_sum(tf.square(lstm1), axis=1))
        s2 = tf.sqrt(tf.reduce_sum(tf.square(lstm2), axis=1))
        s1_s2 = tf.reduce_sum(tf.multiply(lstm1, lstm2), axis=1)
        CScost = (s1_s2 / (s1 * s2)+1) * 2.5 # 要映射到0到5区间内的值

    #### 衡量和标签y的距离
        pearsonr = cal_pearsonr(y,CScost) # 使用pearson相关系数衡量
        dif =tf.square(y - CScost) # 使用MSE衡量
        accuracy = tf.sqrt(tf.reduce_mean(dif))

    with tf.name_scope('streetloss'):
        # 定义街区距离的损失函数
        diff = tf.abs(tf.subtract(lstm1, lstm2), name='err_l1')
        diff = tf.reduce_sum(diff, axis=1)  # 每个样本的每一行求和，求街区距离
        sim = tf.clip_by_value(tf.exp(-1.0 * diff), 1e-7, 1.0 - 1e-7)  # clip是限定下限和上限，大于或者小于的就用对应限度值代替
        loss = tf.square(tf.subtract(sim, tf.clip_by_value(y/5, 1e-7, 1.0 - 1e-7)))  # 计算损失和和目标的距离

        cost = tf.sqrt(tf.reduce_mean(loss))  # 求均值
        truecost = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(sim * 5.0 , y))))  # 不clip下的差距

    # tvars = tf.trainable_variables()  # 得到所有需要优化的变量
    # grads = tf.gradients(loss, tvars) # 计算并截断梯度
    # optimizer = tf.train.AdadeltaOptimizer(learning_rate=lr, epsilon=1e-6)
    # train_op = optimizer.apply_gradients(zip(grads, tvars))

    num_epoch = tf.Variable(0, name='global_step', trainable=False)
    learning_rate = tf.train.exponential_decay(lr, num_epoch,decay_steps, decay_rate, staircase=True)

    # train_op = tf.train.AdadeltaOptimizer(learning_rate).minimize(1.0-pearsonr,global_step=num_epoch)
    train_op = tf.train.AdadeltaOptimizer(learning_rate).minimize(dif,global_step=num_epoch)



    with tf.Session() as sess:
        batch_numbers=[]
        acc_list = []
        sess.run(tf.global_variables_initializer())
        train_input = HelperClass('sts-train',n_steps,n_inputs,em.embeddings_index)
        train_input.initial()

        validation_input = HelperClass('sts-test', n_steps, n_inputs, em.embeddings_index)
        validation_input.initial()

        e_step = 0  # 当前迭代的次数
        while e_step<epoch:
            if e_step % 10 == 0:
                print(e_step,"次迭代训练完毕")
                print("当前学习率为：{}".format(sess.run(learning_rate)))

                # flag, batch_xs1, batch_xs2, batch_ys, mask_xs1, mask_xs2 = validation_input.next_batch(batch_size)
                # acc=sess.run(accuracy,feed_dict={x1:batch_xs1,x2:batch_xs2,y:batch_ys,x1_len:mask_xs1,x2_len:mask_xs2})
                # print(acc)
                # acc_list.append(acc)
                # batch_numbers.append(e_step)

                acc,flag,step=0,1,0
                while flag:
                    # if step%100==0:
                    #     print(step)
                    flag,batch_xs1, batch_xs2, batch_ys, mask_xs1, mask_xs2 = validation_input.next_batch(batch_size)
                    acc+=sess.run(accuracy,feed_dict={x1:batch_xs1,x2:batch_xs2,y:batch_ys,x1_len:mask_xs1,x2_len:mask_xs2})
                    step+=1
                print(acc/step)
                acc_list.append(acc/step)
                batch_numbers.append(e_step)

            e_step += 1
            flag, batch_xs1, batch_xs2, batch_ys, mask_xs1, mask_xs2 = train_input.next_batch(batch_size)
            sess.run(train_op,feed_dict={x1: batch_xs1, x2: batch_xs2, y: batch_ys, x1_len: mask_xs1, x2_len: mask_xs2})

        plt.figure()
        plt.plot(batch_numbers,acc_list)
        plt.show()
        print(acc_list)

if __name__=="__main__":
    run_model()