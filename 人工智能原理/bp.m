clc;clear all;close all;
data=xlsread("数据集.xlsx");
label=data(:,end);
data=data(:,1:end-1);
N=size(data,1);
M=size(data,2);
index=randperm(N);
p=0.8;
input_train=data(index(1:p*N),:)';
out_train=label(index(1:p*N),:)';
input_test=data(index(p*N+1:end),:)';
out_test=label(index(p*N+1:end),:)';

%%   数据归一化
[input_train1,op]=mapminmax(input_train,0,1);
[input_test1]=mapminmax('apply',input_test,op);
out_train1=ind2vec(out_train);
out_test1=ind2vec(out_test);

inputnum=size(data,2);
outputnum=size(out_train1,1);
hiddenum=6;

net=newff(input_train1,out_train1,hiddenum);
net.trainParam.epochs=1000;
net.trainParam.goal=1e-10;
net.trainParam.lr=0.01;
net.trainParam.showWindow=0;

sm=M*hiddenum+hiddenum+hiddenum*outputnum+outputnum;

X=rand(1,sm); 
w1=X(1:M*hiddenum);
B1=X(M*hiddenum+1:M*hiddenum+hiddenum);
w2=X(M*hiddenum+hiddenum+1:M*hiddenum+hiddenum+hiddenum*outputnum);
B2=X(M*hiddenum+hiddenum+hiddenum*outputnum+1:end);

net.IW{1,1}=reshape(w1,hiddenum,M);
net.Lw{2,1}=reshape(w2,outputnum,hiddenum);
net.b{1}=reshape(B1,hiddenum,1);
net.b{2}=B2';

net.trainParam.showWindow=1;

net=train(net,input_train1,out_train1);

T1=sim(net,input_train1);
T2=sim(net,input_test1);

T1=vec2ind(T1);
T2=vec2ind(T2);

[out_train,index1]=sort(out_train);
[out_test,index2]=sort(out_test);

T1=T1(index1);
T2=T2(index2);

error1=(sum((T1==out_train))/(N-M))*100;
error2=(sum((T2==out_test))/M)*100;

