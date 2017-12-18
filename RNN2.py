#!usr/bin/env python
#-*- coding:utf-8 _*-  
""" 
@author:ZhangHui
@file: RNN2.py 
@time: 2017/12/11 
"""
from GetData import GetData
import tensorflow as tf

time_steps = 1
batch_size = 50
input_size = 11
output_size = 1
cell_size = 10
LR = 0.5
def get_batch():
    global time_steps
    filename = 'TrainData.txt'
    xs,ys = GetData(filename,batch_size)
    #原始数据 x->[None,1,11]  y->[None,1]
    #[1,8,5,7,3,1,9,7,1,9,1]   [888]
    return (xs.reshape([-1,time_steps,input_size]),ys.reshape([-1,output_size]))

class LSTMRNN(object):
    #    model = LSTMRNN(time_steps,input_size,output_size,cell_size,batch_size)
    def __init__(self,n_steps,input_size,output_size,cell_size,batch_size):
        self.n_steps = n_steps   #n_steps = 11   号码11位
        self.input_size = input_size   #input_size = 10    每一位存在10种可能
        self.outputs_size = output_size     #output_size = 1   输入1位价格
        self.cell_size = cell_size    #cell_size = 10
        self.batch_size = batch_size     #batch_size = 50
        with tf.name_scope('inputs'):
            self.xs = tf.placeholder(tf.float32,[None,n_steps,input_size],name='xs')   #(-1,11,10)
            self.ys = tf.placeholder(tf.float32,[None,output_size],name='ys')    #(-1,1)
        with tf.variable_scope('in_hidden'):
            self.add_input_layer()
        with tf.variable_scope('LSTM_cell'):
            self.add_cell()
        with tf.variable_scope('out_hidden'):
            self.add_output_layer()
        # with tf.variable_scope('outlayer'):
        #     self.add_reshape_output_layer()
        with tf.name_scope('cost'):
            self.computer_cost()
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(LR).minimize(self.cost)
    def add_input_layer(self,):
        l_in_x = tf.reshape(self.xs,[-1,self.input_size],name='2_2D')   #方便矩阵相乘(-1*1,11)
        Ws_in = self._weight_variable([self.input_size, self.cell_size])   #(11,10)
        bs_in = self._bias_variable([self.cell_size,])     #(10,)
        with tf.name_scope('Wx_plus_b'):
            l_in_y = tf.nn.relu(tf.matmul(l_in_x,Ws_in) + bs_in)    #l_in_y --> (-1*1,10)
        self.l_in_y = tf.reshape(l_in_y,[-1,self.n_steps,self.cell_size],name='2_3D')  #(-1,1,10)

    def add_cell(self):
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.cell_size,forget_bias=1.0,state_is_tuple=True)
        with tf.name_scope('initial_state'):
            self.cell_init_state = lstm_cell.zero_state(self.batch_size,dtype=tf.float32)
        #cell_outputs --> (50, 1, 10)
        #cell_final_state  LSTMStateTuple
        self.cell_outputs,self.cell_final_state = tf.nn.dynamic_rnn(
            lstm_cell,self.l_in_y,initial_state=self.cell_init_state,time_major=False)

    def add_output_layer(self):
        l_out_x = tf.reshape(self.cell_outputs,[-1,self.cell_size],name='2_2D')   #(50,10)
        Ws_out = self._weight_variable([self.cell_size,self.outputs_size])   #(10,1)
        bs_out = self._bias_variable([self.outputs_size,])   #(1,)
        with tf.name_scope('Wx_plus_b'):
            #pred --> (50,1)
            self.pred = tf.nn.relu(tf.matmul(l_out_x,Ws_out) + bs_out)   #(-1,1)

    # def add_reshape_output_layer(self):
    #     l_out_x = tf.reshape(self.pred, [-1, self.n_steps])
    #     Ws_out_ = self._weight_variable([self.n_steps,self.outputs_size])
    #     bs_out_ = self._bias_variable([self.outputs_size, ])
    #     with tf.name_scope('Wx_plus_b_after_reshape'):
    #         #pred --> 50,1
    #         self.pred = tf.reshape(tf.nn.relu(tf.matmul(l_out_x,Ws_out_) + bs_out_),[-1,1,1])

    def computer_cost(self):
        #logit --> [50,1]   target --> [50*1]
        losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [tf.reshape(self.pred, [-1], name='reshape_pred')],
            [tf.reshape(self.ys, [-1], name='reshape_target')],
            [tf.ones([self.batch_size * self.n_steps], dtype=tf.float32)],
            average_across_timesteps=True,
            softmax_loss_function=self.ms_error,
            name='losses'
        )
        # losses = self.ms_error(self.pred,self.ys)
        with tf.name_scope('average_cost'):
            self.cost = tf.div(
                tf.reduce_sum(losses, name='losses_sum'),self.batch_size,
                name='average_cost')
            tf.summary.scalar('cost', self.cost)

    @staticmethod
    def ms_error(labels,logits):
        return tf.square(tf.subtract(labels,logits))
    def _weight_variable(self,shape,name='weights'):
        initializer = tf.random_normal_initializer(mean=0., stddev=1., )
        return tf.get_variable(shape=shape, initializer=initializer, name=name)
    def _bias_variable(self, shape, name='biases'):
        initializer = tf.constant_initializer(0.1)
        return tf.get_variable(name=name, shape=shape, initializer=initializer)

if __name__ == '__main__':
    model = LSTMRNN(time_steps,input_size,output_size,cell_size,batch_size)
    sess = tf.Session()
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs", sess.graph)
    init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(20000):
        x, y = get_batch()
        if i == 0:
            feed_dict = {
                model.xs:x,
                model.ys:y,
            }
        else:
            feed_dict = {
                model.xs:x,
                model.ys:y,
                model.cell_init_state:state
            }
        _,cost,state,pred = sess.run(
            [model.train_op,model.cost,model.cell_init_state,model.pred],
            feed_dict=feed_dict
        )
        if i % 20 == 0:
            print('cost: ',round(cost,4))
            print('ys: ',y[0])
            print('pred: ',pred[0])
            # result = sess.run(merged,feed_dict)
            # writer.add_summary([result,i])