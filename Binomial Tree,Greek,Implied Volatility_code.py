# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 09:30:58 2021
@author: ivy
"""

import numpy as np
#二叉树法为欧式看涨期权定价


def binarytree_europcall(S,K,r,sigma,t,steps,div):
    '''
    S: 标的资产初始价格；
    K: 期权的执行价格；
    r: 年化无风险利率；
    sigma: 标的资产连续复利收益率的标准差；
    t:以年表示的时间长度；
    steps:二叉树的步长。
    div:股息率
    '''
    u = np.exp((r-div)*t/steps+sigma*np.sqrt(t/steps))#时间间隔为△t=t/steps
    d = np.exp((r-div)*t/steps-sigma*np.sqrt(t/steps))

    
    P = (np.exp((r-div)*t/steps)-d)/(u-d)
    prices = np.zeros(steps+1)
    c_values = np.zeros(steps+1)
    
    prices[0] = S*d**steps #生成最后一列的股票价格空数组
    c_values[0] = np.maximum(prices[0]-K,0)#生成最后一列的期权价值空数组
    for i in range(1,steps+1):
        prices[i] = prices[i-1]*u/d   #计算最后一列的股票价格
        #print(prices[i])
        c_values[i] = np.maximum(prices[i]-K,0)#计算最后一列的期权价值
        #print(c_values[i])
    for j in range(steps,0,-1):#逐个节点往前计算
        for i in range(0,j):
            c_values[i] = (P*c_values[i+1]+(1-P)*c_values[i])*np.exp(-r*t/steps)
         
    return c_values[0]

A1=binarytree_europcall(130,247,0.07,0.35,8/12,8,0.02)
print('二叉树法为欧式看涨期权定价：',round(A1,2))



def Value_European(S,K1,K2,r,sigma,t,div,T1,T2,step1,step2):
    '''
    S： 股票当当前价格
    K1： 外层期权的执行价格
    K2: 内层期权的执行价格
    r: 无风险利率
    sigma:股票连续复利收益率的标准差；
    t:以年表示的总时间长度
    div:股息率
    T1：外层期权的到期期限
    T2：内层期权的到期期限
    step1: 外层期权二叉树的步数
    step2: 内层期权二叉树的步数
    
    '''
    
    step= step1+step2 #将总步数分为外层期权的步数step1和欧式期权步数step2
    u = np.exp((r-div)*t/step+sigma*np.sqrt(t/step)) #时间间隔为△t=t/steps
    d = np.exp((r-div)*t/step-sigma*np.sqrt(t/step))

    P = (np.exp((r-div)*t/step)-d)/(u-d)
    
    c_t1_values=np.zeros(step1+1)
    v_values=np.zeros(step1+1)
    
    for i in range(0,step1+1):
        c_t1_values[i]=binarytree_europcall(S*(u**i)*(d**(step1-i)),K2,r,sigma,T2,step2,div)
       # print(c_t1_values[i])
        v_values[i]=np.maximum(c_t1_values[i]-K1,0)
     
    for j in range(step1,0,-1):#逐个节点往前计算
        for i in range(0,j):
            v_values[i] = (P*v_values[i+1]+(1-P)*v_values[i])*np.exp(-r*T1/step1)
            
    return v_values[0]    


def Value_American(S,K1,K2,r,sigma,t,div,T1,T2,step1,step2):
    
    step= step1+step2
    u = np.exp((r-div)*t/step+sigma*np.sqrt(t/step))#时间间隔为△t=t/steps
    d = np.exp((r-div)*t/step-sigma*np.sqrt(t/step))
    P = (np.exp((r-div)*t/step)-d)/(u-d)
    
    c_t1_values=np.zeros(step1+1)
    v_values=np.zeros(step1+1)
    
    for i in range(0,step1+1):
        c_t1_values[i]=binarytree_europcall(S*(u**i)*(d**(step1-i)),K2,r,sigma,T2,step2,div)
       # print(c_t1_values[i])
        v_values[i]=np.maximum(c_t1_values[i]-K1,0)
       # print(c_t1_values[i])
     
    for j in range(step1,0,-1):#逐个节点往前计算
        for i in range(0,j):
            c_t1_values[i]=binarytree_europcall(S*(u**i)*(d**(step1-1-i)),K2,r,sigma,T2+j*(T1/step1),step2+j,div)
            #print(c_t1_values[i])
            v_values[i] = np.maximum((P*v_values[i+1]+(1-P)*v_values[i])*np.exp(-r*T1/step1),c_t1_values[i]-K1)
            
    return v_values[0]    


#-----------------------------------------------------------------------
#change the parameters
#initial
rr=0.0291#riskfree-rate
div_ICBC=0.06 #dividend yield of ICBC
sigma_ICBC=0.221843
S0=4.27
K1=0.1
K2=5
T1=0.5
T2=0.5
steps=120
T=T1+T2
h=1/120

#list of change
K1_list=[0.02,0.04,0.06,0.08,0.1,0.12,0.14,0.16,0.18,0.2]
K2_list=[4.2, 4.4, 4.6, 4.8, 5, 5.2, 5.4, 5.6, 5.8, 6]
T1_list=[1/12,2/12,3/12,6/12,9/12,12/12]
T2_list=[1/12,2/12,3/12,6/12,9/12,12/12]
step1_list=[10,20,30,60,90,120]
step2_list=[10,20,30,60,90,120]
N_list=[10,20,50,100,200,500,1000]
N_half=[5,10,25,50,100,250,500]# number of steps in T1 and T2, because initial T1=T2


#1.change T1

v_change_T1=np.zeros(np.size(T1_list))
print("change T1")
print("European")
for i in range(0,np.size(T1_list)):
    v_change_T1[i]=Value_European(S=S0,K1=K1,K2=K2,r=rr,sigma=sigma_ICBC,t=T1_list[i]+T2,div=div_ICBC,T1=T1_list[i],T2=T2,step1=step1_list[i],step2=60)
    print(v_change_T1[i])

print("American")
for i in range(0,np.size(T1_list)):
    v_change_T1[i]=Value_American(S=S0,K1=K1,K2=K2,r=rr,sigma=sigma_ICBC,t=T1_list[i]+T2,div=div_ICBC,T1=T1_list[i],T2=T2,step1=step1_list[i],step2=60)
    print(v_change_T1[i])
    
    
    
#2.change T2
print("change T2")
print("European")
v_change_T2=np.zeros(np.size(T2_list))
for i in range(0,np.size(T2_list)):
    v_change_T2[i]=Value_European(S=S0,K1=K1,K2=K2,r=rr,sigma=sigma_ICBC,t=T1+T2_list[i],div=div_ICBC,T1=T1,T2=T2_list[i],step1=60,step2=step2_list[i])
    print(v_change_T2[i])

print("American")
for i in range(0,np.size(T2_list)):
    v_change_T2[i]=Value_American(S=S0,K1=K1,K2=K2,r=rr,sigma=sigma_ICBC,t=T1+T2_list[i],div=div_ICBC,T1=T1,T2=T2_list[i],step1=60,step2=step2_list[i])
    print(v_change_T2[i])

#3.change K1
print("change K1")
print("European")
v_change_K1=np.zeros(np.size(K1_list))
for i in range(0,np.size(K1_list)):
    v_change_K1[i]=Value_European(S=S0,K1=K1_list[i],K2=K2,r=rr,sigma=sigma_ICBC,t=T1+T2,div=div_ICBC,T1=T1,T2=T2,step1=60,step2=60)
    print(v_change_K1[i])
    
print("American")
for i in range(0,np.size(K1_list)):
    v_change_K1[i]=Value_American(S=S0,K1=K1_list[i],K2=K2,r=rr,sigma=sigma_ICBC,t=T1+T2,div=div_ICBC,T1=T1,T2=T2,step1=60,step2=60)
    print(v_change_K1[i])
    
    
#4.change K2
print("change K2")
print("European")
v_change_K2=np.zeros(np.size(K2_list))
for i in range(0,np.size(K2_list)):
    v_change_K2[i]=Value_European(S=S0,K1=K1,K2=K2_list[i],r=rr,sigma=sigma_ICBC,t=T1+T2,div=div_ICBC,T1=T1,T2=T2,step1=60,step2=60)
    print(v_change_K2[i])
    
print("American")
for i in range(0,np.size(K2_list)):
    v_change_K2[i]=Value_American(S=S0,K1=K1,K2=K2_list[i],r=rr,sigma=sigma_ICBC,t=T1+T2,div=div_ICBC,T1=T1,T2=T2,step1=60,step2=60)
    print(v_change_K2[i])
    

#5.change N
print("change N")
print("European")
v_change_N=np.zeros(np.size(N_list))
for i in range(0,np.size(N_list)):
    v_change_N[i]=Value_European(S=S0,K1=K1,K2=K2,r=rr,sigma=sigma_ICBC,t=T1+T2,div=div_ICBC,T1=T1,T2=T2,step1=N_half[i],step2=N_half[i])
    print(v_change_N[i])
    
print("American")
for i in range(0,np.size(N_list)):
    v_change_N[i]=Value_American(S=S0,K1=K1,K2=K2,r=rr,sigma=sigma_ICBC,t=T1+T2,div=div_ICBC,T1=T1,T2=T2,step1=N_half[i],step2=N_half[i])
    print(v_change_N[i])


#----------------------------------------------------------------------------------
#求隐含波动率（二分法）
#开始时间为2021年11月1日，根据市场上现存的期权我们得出到期日T2_ICBC列表



real_price=0.084#市场中的真实期权价格

from scipy.stats import norm
N = norm.cdf

def bs_call(S, K, T, r,div, vol):
    d1 = (np.log(S/K) + (r -div+ 0.5*vol**2)*T) / (vol*np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    return S * norm.cdf(d1)*np.exp(-div*T) - np.exp(-r * T) * K * norm.cdf(d2)

def bs_vega(S, K, T, r,div, sigma):
    d1 = (np.log(S / K) + (r -div+ 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)

def find_vol(target_value, S, K, T, r, *args):
    MAX_ITERATIONS = 200
    PRECISION = 1.0e-5
    sigma = 0.221843
    div=0.061732
    for i in range(0, MAX_ITERATIONS):
        price = bs_call(S, K, T, r,div,sigma)
        vega = bs_vega(S, K, T, r, div,sigma)
        diff = target_value - price  # our root
        if (abs(diff) < PRECISION):
            return sigma
        sigma = sigma + diff/vega # f(x) / f'(x)
    return sigma # value wasn't found, return best guess so far

implied_vol=find_vol(0.084, 4.27, 5, 10/12, 0.0291)
implied_c=binarytree_europcall(S=4.27,K=5,r=0.0291,sigma=implied_vol,t=10/12,steps=60,div=0.061732)


print("implied_vol:",implied_vol)
print("带入隐含波动率：" ,implied_c)
#---------------------------------------------------------------------------------------
#calculate Greek


sigma=0.221843
r=0.0291
div=0.061732
S0=4.27
T1=0.5
T2=0.5
K1=0.1
K2=5
step1=60
step2=60
t=T1+T2
steps=step1+step2

u = np.exp((r-div)*t/steps+sigma*np.sqrt(t/steps))#时间间隔为△t=t/steps
d = np.exp((r-div)*t/steps-sigma*np.sqrt(t/steps))
h=t/steps

v=Value_European(S=S0,K1=K1,K2=K2,r=r,sigma=sigma,t=t,div=div,T1=T1,T2=T2,step1=step1,step2=step2)


va=Value_American(S=S0,K1=K1,K2=K2,r=r,sigma=sigma,t=t,div=div,T1=T1,T2=T2,step1=step1,step2=step2)



def delta1(s,n):
    vu=Value_European(S=s*u,K1=K1,K2=K2,r=r,sigma=sigma,t=t-(n+1)*T1/step1,div=div,T1=T1-(n+1)*T1/step1,T2=T2,step1=step1-n-1,step2=step2)
    vd=Value_European(S=s*d,K1=K1,K2=K2,r=r,sigma=sigma,t=t-(n+1)*T1/step1,div=div,T1=T1-(n+1)*T1/step1,T2=T2,step1=step1-n-1,step2=step2)
    delta1=np.exp(-div*(t/steps))*(vu-vd)/(s*u-s*d)
    return delta1


print(delta1(u*S0,1),delta1(d*S0,1))


gamma1=(delta1(u*S0,1)-delta1(d*S0,1))/(u*S0-d*S0)

ee=u*d*S0-S0
v_uds=Value_European(S=S0*u*d,K1=K1,K2=K2,r=r,sigma=sigma,t=t-2*h,div=div,T1=T1-2*h,T2=T2,step1=step1-2,step2=step2)
theta1=(v_uds-ee*delta1(S0,0)-1/2*ee**2*gamma1-v)/(2*h)


def delta2(s,n):
    vua=Value_American(S=s*u,K1=K1,K2=K2,r=r,sigma=sigma,t=t-(n+1)*T1/step1,div=div,T1=T1-(n+1)*T1/step1,T2=T2,step1=step1-n-1,step2=step2)
    vda=Value_American(S=s*d,K1=K1,K2=K2,r=r,sigma=sigma,t=t-(n+1)*T1/step1,div=div,T1=T1-(n+1)*T1/step1,T2=T2,step1=step1-n-1,step2=step2)
    delta2=np.exp(-div*(t/steps))*(vua-vda)/(s*u-s*d)
    return delta2
gamma2=(delta2(u*S0,1)-delta2(d*S0,1))/(u*S0-d*S0)
ee=u*d*S0-S0
va_uds=Value_American(S=S0*u*d,K1=K1,K2=K2,r=r,sigma=sigma,t=t-2*h,div=div,T1=T1-2*h,T2=T2,step1=step1-2,step2=step2)
theta2=(va_uds-ee*delta2(S0,0)-1/2*ee**2*gamma1-va)/(2*h)





print(u,d)
print("European situation")
print("delta=",delta1(S0,0))
print("gamma=",gamma1)
print("theta=",theta1,"\n")

print("American situation")
print("delta=",delta2(S0,0))
print("gamma=",gamma2)
print("theta=",theta2,"\n")


def uds(s):
    return Value_European(S=s*u*d,K1=K1,K2=K2,r=r,sigma=sigma,t=t-2*h,div=div,T1=T1-2*h,T2=T2,step1=step1-2,step2=step2)
    
def gamma(s):
    return (delta1(u*s,1)-delta1(d*s,1))/(u*s-d*s)

def v(s):
    return Value_European(s,K1=K1,K2=K2,r=r,sigma=sigma,t=t,div=div,T1=T1,T2=T2,step1=step1,step2=step2)


def udsa(s):
    return Value_American(S=s*u*d,K1=K1,K2=K2,r=r,sigma=sigma,t=t-2*h,div=div,T1=T1-2*h,T2=T2,step1=step1-2,step2=step2)
    
def gammaa(s):
    return (delta2(u*s,1)-delta2(d*s,1))/(u*s-d*s)

def va(s):
    return Value_American(s,K1=K1,K2=K2,r=r,sigma=sigma,t=t,div=div,T1=T1,T2=T2,step1=step1,step2=step2)





import matplotlib.pyplot as plt 

x = [4.2, 4.4, 4.6, 4.8, 5, 5.2, 5.4, 5.6, 5.8, 6]
y1=np.zeros(np.size(x))
y2 =np.zeros(np.size(x))

for i in range(0,np.size(x)):
    y1[i]=(delta1(u*x[i],1)-delta1(d*x[i],1))/(u*x[i]-d*x[i])

for i in range(0,np.size(x)):
    y2[i]=(delta2(u*x[i],1)-delta2(d*x[i],1))/(u*x[i]-d*x[i])
  


d1=np.zeros(np.size(x))
d2=np.zeros(np.size(x))
for i in range(0,np.size(x)):
    d1[i]=delta1(x[i],0)
    d2[i]=delta2(x[i],0)
 

t1=np.zeros(np.size(x))
t2=np.zeros(np.size(x))
for i in range(0,np.size(x)):
    t1[i]=(uds(x[i])-ee*delta1(S0,0)-1/2*ee**2*gamma(x[i])-v(x[i]))/(2*h)
    t2[i]=(udsa(x[i])-ee*delta2(S0,0)-1/2*ee**2*gammaa(x[i])-va(x[i]))/(2*h)

plt.plot(x, t1,label='European') 
plt.plot(x, t2,label='American') 
plt.title('line chart') 
plt.xlabel('s') 
plt.ylabel('Theta') 
plt.legend(loc=0,ncol=1)
plt.show()

