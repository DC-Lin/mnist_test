import numpy as np
import visdom
viz=visdom.Visdom()
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))
def dsigmoid(z):
    return sigmoid(z)*(1-sigmoid(z))

class MLP:
    def __init__(self,sizes):
        '''
        [28*28,30,10]
        :param sizes:
        w1=784*30
        w2=30*10
        b1=30
        b2=10
        '''
        self.sizes=sizes
        self.weights=[np.random.randn(c2,c1) for c1,c2 in zip(sizes[:-1],sizes[1:])]
        self.biases=[np.random.randn(ch,1) for ch in sizes[1:]]

    def forward(self,x):
        '''
        x=[784,1] sigle chanel
        :param x:
        :return:
        '''
        for b,w in zip(self.biases,self.weights):
            z=np.dot(w,x)+b
            x=sigmoid(z)
        return x
    def backprop(self,x,y):
        '''
        x=[784,1]
        y=[10,1]
        :param x:
        :param y:
        :return:
        '''
        nabla_w=[np.zeros(w.shape) for w in self.weights]
        nabla_b=[np.zeros(b.shape) for b in self.biases]
        activations=[x]
        zs=[]
        activation=x
        for b,w in zip(self.biases,self.weights):
            z=np.dot(w,activation)+b
            activation=sigmoid(z)
            zs.append(z)
            activations.append(activation)
        loss=np.power(activations[-1]-y,2).sum()
        #compute gradient on out layer
        delta=activations[-1]*(1-activations[-1])*(activations[-1]-y)
        nabla_b[-1]=delta
        #矩阵相乘，每个最终输出都与上一个层的单个神经元有连接，每条线都是一个w，矩阵相乘记录每个输出神经元与上一层每个神经元的w梯度
        nabla_w[-1]=np.dot(delta,activations[-2].T)
        #compute hidden gradient适用于多隐藏层，因为delta存在.
        for l in range(2,len(self.sizes)):
            l=-l
            z=zs[l]
            delta=np.dot(self.weights[l+1].T,delta)*activations[l]*(1-activations[l])
            nabla_b[l]=delta
            nabla_w[l]=np.dot(delta,activations[l-1].T)
        return nabla_w,nabla_b,loss
    def train(self,train_data,epochs,batch_size,lr,test_data):
        '''

        :param train_data:(x,y)
        :param epochs:
        :param batch_size:
        :param lr:
        :param test_data:
        :return:
        '''
        if test_data:
            n_test=len(test_data)
        n=len(train_data)
        import random
        for epoch in range(epochs):
            random.shuffle(train_data)
            mini_batches=[train_data[k:k+batch_size] for k in range(0,n,batch_size)]
            for mini_batche in mini_batches:
                loss=self.update_mini_batch(mini_batche,lr)
            if test_data:
                print(f'epoch{epoch}:{self.evaluate(test_data)},loss:{loss},{n_test}')
            else:
                print(f'epoch{epoch}completed')
            viz.line([loss],[epoch],win='loss-vis',update='append',opts=dict(title='loss-vis'))

    def update_mini_batch(self,batch,lr):
        '''

        :param batch:
        :param lr:
        :return:
        '''
        nabla_w=[np.zeros(w.shape) for w in self.weights]
        nabla_b=[np.zeros(b.shape) for b in self.biases]
        for x,y in batch:
            gra_w,gra_b,loss=self.backprop(x,y)
            nabla_w=[accu+cur for accu,cur in zip(nabla_w,gra_w)]
            nabla_b=[accu+cur for accu,cur in zip(nabla_b,gra_b)]
            loss+=loss
        #batch Norm
        nabla_w=[w/len(batch) for w in nabla_w]
        nabla_b=[b/len(batch) for b in nabla_b]
        loss/=len(batch)
        self.weights=[w-lr*nabla for w,nabla in zip(self.weights,nabla_w)]
        self.biases=[b-lr*nabla for b,nabla in zip(self.biases,nabla_b)]
        return loss
    def evaluate(self,test_data):
        '''
        :param test_data:
        :return:
        '''
        result=[(np.argmax(self.forward(x)),y) for x,y in test_data]
        correct=sum(int(pred==y) for pred,y in result)
        return correct

def main():
    import dataloader
    tra_data,val_data,test_data=dataloader.load_data_wrapper()
    mlp=MLP([784,30,10])
    mlp.train(tra_data,100,10,1e-3,test_data=test_data)

if __name__ == '__main__':
    main()
