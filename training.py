import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    z=z.flatten()
    idx_zero=np.where(z<0)[0]
    idx_one=np.where(z>=0)[0]
    result=np.zeros((len(z),))
    result[idx_one]=1/(1+np.exp(-z[idx_one]))
    result[idx_zero]=np.exp(z[idx_zero])/(1+np.exp(z[idx_zero]))
    return result.reshape(len(z),1)


def training(x,y_actual,learning_rate,epochs):
    loss=[]
    y_actual=np.array(y_actual).reshape(len(y_actual),1)
    w=np.zeros((len(x[0]),1))
    b=0
    prev=0
    now=0
    count=0

    while count<=epochs:
        z=x@w+b
        y_pred=sigmoid(z)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        m=y_actual*np.log(y_pred)
        n=(1-y_actual)*np.log(1-y_pred)
        J=-(1/len(x))*np.sum(m+n)
        m=-1/len(x)
        n=y_actual-y_pred
        der_w=m*np.transpose(x)@n
        der_b=m*np.sum(n)
        w=w-der_w*learning_rate
        b=b-der_b*learning_rate
        loss.append(J)
        prev=now
        now=J
        if count%50==0:
            print("Epoch:",count,"      Loss: ",J)
        count+=1
        
    print("Total Epochs:",count,"       Final Loss:",now)
    epochs=np.arange(count)
    plot_graph(epochs,loss)
    return w,b


def plot_graph(epochs,loss):
    plt.plot(epochs,loss,color="blue",linewidth=2,linestyle="-")
    plt.title("Loss Graph")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Loss")
    plt.grid(linestyle="--",color="grey")
    plt.savefig("loss curve.png",dpi=300)
    plt.show()