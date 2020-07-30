#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import minmax_scale
from ta import *
import datetime
from datetime import datetime
import os
import glob


data_list = []
data_list_x = []
data_list_y = []

for filename in glob.glob(os.path.join('*.csv')):
    df = pd.read_csv(filename)
    data_list.append(df)
    data_list_x.append(df.loc[0:len(df.index), ['Open','High','Low','Close']])
    data_list_y.append(df.loc[0:len(df.index), ['Difference']])
    # break

# print(data_list_y)




# In[2]:




#Stockdata = pd.read_csv("AA_New_Extracolumn.csv",sep=',')
#df = pd.DataFrame(Stockdata)
# we need to convert from string date to integer from 2001 1 1 in order to handle NaNs
# import datetime
# dateFrom = datetime.datetime(2001,1,1)
# from datetime import datetime
# for i in data_list:
#     for j in range(0, len(i['Timestamp'])):
#         i.loc[j, ('Timestamp')] = (datetime.strptime(df.loc[j, ('Timestamp')], '%m/%d/%Y') - dateFrom).days


# In[3]:


# Clean NaN values
# df = df.fillna("")
# df = df.dropna()


# In[4]:


#We added a very small number to division in his function to avoid divide by zero error
def adx(high, low, close, n=14, fillna=False):
    """Average Directional Movement Index (ADX)

    The Plus Directional Indicator (+DI) and Minus Directional Indicator (-DI)
    are derived from smoothed averages of these differences, and measure trend
    direction over time. These two indicators are often referred to
    collectively as the Directional Movement Indicator (DMI).

    The Average Directional Index (ADX) is in turn derived from the smoothed
    averages of the difference between +DI and -DI, and measures the strength
    of the trend (regardless of direction) over time.

    Using these three indicators together, chartists can determine both the
    direction and strength of the trend.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:average_directional_index_adx

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    cs = close.shift(1)
    pdm = high.combine(cs, lambda x1, x2: get_min_max(x1, x2, 'max'))
    pdn = low.combine(cs, lambda x1, x2: get_min_max(x1, x2, 'min'))
    tr = pdm - pdn

    trs_initial = np.zeros(n-1)
    trs = np.zeros(len(close) - (n - 1))
    trs[0] = tr.dropna()[0:n].sum()
    tr = tr.reset_index(drop=True)
    for i in range(1, len(trs)-1):
        trs[i] = trs[i-1] - (trs[i-1]/float(n)) + tr[n+i]

    up = high - high.shift(1)
    dn = low.shift(1) - low
    pos = abs(((up > dn) & (up > 0)) * up)
    neg = abs(((dn > up) & (dn > 0)) * dn)

    dip_mio = np.zeros(len(close) - (n - 1))
    dip_mio[0] = pos.dropna()[0:n].sum()

    pos = pos.reset_index(drop=True)
    for i in range(1, len(dip_mio)-1):
        dip_mio[i] = dip_mio[i-1] - (dip_mio[i-1]/float(n)) + pos[n+i]

    din_mio = np.zeros(len(close) - (n - 1))
    din_mio[0] = neg.dropna()[0:n].sum()

    neg = neg.reset_index(drop=True)
    for i in range(1, len(din_mio)-1):
        din_mio[i] = din_mio[i-1] - (din_mio[i-1]/float(n)) + neg[n+i]

    dip = np.zeros(len(trs))
    for i in range(len(trs)):
        dip[i] = 100 * (dip_mio[i]/(trs[i] + 0.00000000000001))

    din = np.zeros(len(trs))
    for i in range(len(trs)):
        din[i] = 100 * (din_mio[i]/(trs[i] + 0.00000000000001))

    dx = 100 * np.abs((dip - din) / ((dip + din) + 0.00000000000001))

    adx = np.zeros(len(trs))
    adx[n] = dx[0:n].mean()

    for i in range(n+1, len(adx)):
        adx[i] = ((adx[i-1] * (n - 1)) + dx[i-1]) / float(n)

    adx = np.concatenate((trs_initial, adx), axis=0)
    adx = pd.Series(data=adx, index=close.index)

    if fillna:
        adx = adx.replace([np.inf, -np.inf], np.nan).fillna(20)
    return pd.Series(adx, name='adx')


# In[5]:


# Add ta features filling NaN values
# df = add_all_ta_features(df, "Open", "High", "Low", "Close", "Volume", fillna=True)


# In[3]:


def normalize_list_numpy(list_numpy):
    normalized_list = minmax_scale(list_numpy)
    return normalized_list


# In[ ]:

data_list[len(data_list)-5]['Prediction'] = 0
predictions = []

def write_predictions(pred):
	for i in range (0,len(pred)):
		data_list[len(data_list)-5].loc[i+20, 'Prediction'] = pred[i]
	data_list[len(data_list)-5].to_csv("VTR_New_Pred_All.csv")


def sample_points(k,start):
    
    choice = random.randint(0,len(data_list_x)-6)

    x = data_list_x[choice].loc[start:start+k-1]
    y = data_list_y[choice].loc[start:start+k-1]

    x1 = np.array(x.values.tolist())
    y1 = np.array(y.values.tolist())

    x1 = (x1 - np.mean(x1)) / np.std(x1)
    
    return (x1, y1)

x, y = sample_points(5,0)
print (x)
print (y)


def sample_points_test(k,start):

    x = data_list_x[len(data_list)-5].loc[start:start+k-1]
    y = data_list_y[len(data_list)-5].loc[start:start+k-1]

    x1 = np.array(x.values.tolist())
    y1 = np.array(y.values.tolist())

    return (x1,y1)


tf.reset_default_graph()


num_hidden = 60
num_classes = 1
num_feature = 4 #5
beta = 0.01

W = tf.Variable(tf.zeros([num_feature, 1]))
b = tf.Variable(tf.zeros(1))

X = tf.placeholder(tf.float32, shape=[None, num_feature])
Y = tf.placeholder(tf.float32, shape=[None, num_classes])




#w2 = tf.Variable(tf.random_uniform([num_hidden, num_classes]))
#b2 = tf.Variable(tf.random_uniform([num_classes]))

#layer 1
#z1 = tf.matmul(X, w1) + b1
#a1 = tf.nn.tanh(z1)


#output layer
#z2 = tf.matmul(a11, w2) + b2
Yhat = tf.matmul(X, W) + b
#Yhat = tf.nn.tanh(z2)


lossfunction = tf.reduce_mean(tf.square(Yhat - Y))
regularizer = tf.nn.l2_loss(W)
lossfunction = tf.reduce_mean(lossfunction + beta * regularizer)


optimizer = tf.train.AdamOptimizer(1e-2).minimize(lossfunction)

init = tf.global_variables_initializer()

#number of epochs i.e training iterations
num_epochs = 60  #4307


#number of samples i.e number of shots
num_samples = 5   #10

#number of task0s
num_tasks = len(data_list) - 5

#number of times we want to perform optimization
num_iterations = 300    #300

#mini btach size
mini_batch = 5

losses = []

#start the tensorflow session
with tf.Session() as sess:

    sess.run(init)

    for p in range(0,50,5):

        for e in range(num_epochs):
            
            #for each task in batch of tasks
            for task in range(num_tasks):

                #get the initial parameters of the model
                old_W, old_b = sess.run([W, b])

                #sample x and y
                x_sample, y_sample = sample_points(num_samples,p)


                #for some k number of iterations perform optimization on the task
                for k in range(num_iterations):

                    #get the minibatch x and y
                    for i in range(0, num_samples, mini_batch):

                        #sample mini batch of examples 
                        x_minibatch = x_sample[i:i+mini_batch]
                        y_minibatch = y_sample[i:i+mini_batch]


                        train = sess.run(optimizer, feed_dict={X: x_minibatch.reshape(mini_batch,4), 
                                                            Y: y_minibatch.reshape(mini_batch,1)})

                #get the updated model parameters after several iterations of optimization
                new_W, new_b = sess.run([W, b])

                #Now we perform meta update

                #i.e theta = theta + epsilon * (theta_star - theta)

                epsilon = 0.1

                updated_W = old_W + epsilon * (new_W - old_W) 
                updated_b = old_b + epsilon * (new_b - old_b) 

                #updated_w2 = old_w2 + epsilon * (new_w2 - old_w2) 
                #updated_b2 = old_b2 + epsilon * (new_b2 - old_b2) 


                #update the model parameter with new parameters
                W.load(updated_W, sess)
                b.load(updated_b, sess)

                #w2.load(updated_w2, sess)
                #b2.load(updated_b2, sess)

            if e%10 == 0:
                loss = sess.run(lossfunction, feed_dict={X: x_sample.reshape(num_samples,4), Y: y_sample.reshape(num_samples,1)})
                losses.append(loss)

                print ("Epoch {}: Loss {}\n".format(e,loss) )            
                print ('Updated Model Parameter Theta\n')
                print ('Sampling Next Batch of Tasks \n')
                print ('---------------------------------\n')
                

        print ('Test ---------------------------------\n')		
        #for each task in batch of tasks
        for task in range(1):

            #get the initial parameters of the model
            old_W, old_b = sess.run([W, b])

            #sample x and y
            x_sample, y_sample = sample_points_test(num_samples,p)


            #for some k number of iterations perform optimization on the task
            for k in range(num_iterations):

                #get the minibatch x and y
                for i in range(0, num_samples, mini_batch):

                    #sample mini batch of examples 
                    x_minibatch = x_sample[i:i+mini_batch]
                    y_minibatch = y_sample[i:i+mini_batch]


                    train = sess.run(optimizer, feed_dict={X: x_minibatch.reshape(mini_batch,4), 
                                                        Y: y_minibatch.reshape(mini_batch,1)})

            #get the updated model parameters after several iterations of optimization
            new_W, new_b = sess.run([W, b])

            #Now we perform meta update

        loss = sess.run(lossfunction, feed_dict={X: x_sample.reshape(num_samples,4), Y: y_sample.reshape(num_samples,1)})
        y_pred = sess.run(Yhat, feed_dict={X: x_sample.reshape(num_samples, 4)})
        print('y_pred -----------------')
        print(y_pred.reshape(1,num_samples))
        print('y -----------------')
        print(y_sample.reshape(1,num_samples))	
        print ("Epoch {}: Test Loss {}\n".format(1,loss) )
        for i in range (0,y_pred.size):
            predictions.append(y_pred[i,0])

    write_predictions(predictions)       

lossdata = {'Loss':losses}
losslist = []
for i in range (0,len(losses)):
    losslist.append(((i%6)+1)*10)
df = pd.DataFrame(lossdata,index=losslist)
df.to_csv("VTR_Loss_All.csv")

# In[ ]:




