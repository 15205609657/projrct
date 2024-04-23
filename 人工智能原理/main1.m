%%  清空环境变量          系统操作
warning off             % 关闭报警信息
close all               % 关闭开启的图窗
clear                   % 清空变量
clc                     % 清空命令行
tic
%%  导入数据
% Data = load("iris.mat");
% data=Data.feat;
% data(:,5)=Data.label;
% for i=1:size(data,1)
%     if data(i,5)==0
%         data(i,5)=3;
%     end
% end
% data=load("heartstatlog\heart.dat");

data=xlsread("数据集.xlsx");       %读取数据
N=size(data,1);     %求出数据集的个数（有多少行）
dim=size(data,2);   %求出数据集的维度，一般一列就是一个特征。
rate=0.9;           %设定划分样本的数量，百分之九十划分训练集，百分之十划分测试集
M=N*0.9;            %训练集的数量
%%  划分训练集和测试集
temp = randperm(N);     %使用函数给样本重新排顺序，按行打乱，防止相同的类别都在一起

input_train = data(temp(1:M), 1: end-1)';   %取dada（数据集）的前1-M行，前1到倒数第二列作为训练集的输入（打开数据集表可以看到只有最后一列是分类的属性），最后把矩阵进行转置（'）,因为神经网络必须转置。
output_train = data(temp(1:M), end)';       %取dada（数据集）的前1-M行，最后一列作为训练集的输出（因为只有最后一列是做的分类），最后转置。
m = size(input_train, 2);   %训练集样本数量（取input_train的行数，正常的数据行是样本数，但是input_train进行转置，所以取他的列数）;其实M,m值基本上相等。

input_test = data(temp(M+1:end), 1: end-1)';    %和上面一样的操作。
output_test = data(temp(M+1:end), end)';
n = size(input_test, 2);    %测试集样本数量

%%  数据归一化       就是把数据控制在一个范围内，防止越界。
[input_train1, ps_input] = mapminmax(input_train, 0, 1);    %调用函数控制在（0，1）内,input_train作为函数的输入，输出input_train1是归一化后的，ps_input就是一个模板，以后直接调用这个模板。就可以进行相同的归一化操作,下一行就是用了
input_test1  = mapminmax('apply', input_test, ps_input);    %应用上面相同的模板，把input_test（测试集输入）进行同样的归一化；
%因为input_train和input_test是矩阵，而output_train和output_test是向量，所以归一化调用的函数不一样（个人理解，我也不是很懂）
output_train1 = ind2vec(output_train);
output_test1  = ind2vec(output_test );

%%  节点个数
inputnum  = size(input_train1, 1);  % 输入层节点数    表的每行作为权值和阈值进行输入
hiddennum = 6;                 % 隐藏层节点数            
outputnum = size(output_test1, 1);  % 输出层节点数  

%%智能优化算法的初始化

sizepop=5;     %种群的数量
maxgen  =   300;        % 种群更新次数  
Vmax    =  1.0;        % 最大速度
Vmin    = -1.0;        % 最小速度
popmax  =  2.0;        % 最大边界
popmin  = -2.0;        % 最小边界

%%神经网络
numsum = inputnum * hiddennum + hiddennum + hiddennum * outputnum + outputnum;  %计算节点总数，就是权重和阈值的总数量。隐含层为一层

%%智能优化算法的初始化
for i = 1 : sizepop
    pop(i, :) = rands(1, numsum);  % 初始化种群
    V(i, :) = rands(1, numsum);    % 初始化速度
end

%%通过调用智能优化算法找出最优的权值和阈值的组合，使得测试集的正确率尽可能地高
[zbest,bestfit,net]=PSO(pop,V,maxgen,hiddennum,input_train1, output_train1,inputnum,outputnum,popmin,popmax,Vmin,Vmax);
[zbest1,bestfit1,net1]=EGPSO(pop,V,maxgen,hiddennum,input_train1, output_train1,inputnum,outputnum,popmin,popmax,Vmin,Vmax);
%   zbest是经过智能优化算法求出来的目前最优的权值和阈值组合成的向量，bestfit是最优值，net是用zbest训练好的网络
%   第一行和第二行是两个不同的智能优化算法     

%%进行仿真，用网络求出该网络得到的训练集和测试集的输出
train_sim=sim(net,input_train1); %训练集的结果
test_sim=sim(net,input_test1);  %测试集的结果
train_sim1=sim(net1,input_train1);
test_sim1=sim(net1,input_test1);


Train_sim=vec2ind(train_sim);%  反归一化，之前归一化了，后面要进行对比是否和真实的训练集、测试集相等，所以必须还原。
Test_sim=vec2ind(test_sim);
Train_sim1=vec2ind(train_sim1);
Test_sim1=vec2ind(test_sim1);

[output_train,index1]=sort(output_train);%得到真实的训练集的分类结果，并且按照一定的顺序排序，坐标序列返回到index1
[output_test,index2]=sort(output_test);%得到真实的测试集的分类结果

%%这里是使用index1把训练出来的训练集分类结果和测试集分类结果和上面的排序弄成一样，要不然可能会错位。
Train_sim=Train_sim(index1);
Test_sim=Test_sim(index2);
Train_sim1=Train_sim1(index1);
Test_sim1=Test_sim1(index2);

%%统计    有多少正确的
error1=sum((Train_sim==output_train))/m*100;
error2=sum((Test_sim==output_test))/n*100;
error3=sum((Train_sim1==output_train))/m*100;
error4=sum((Test_sim1==output_test))/n*100;

%%下面是画图部分
figure
subplot(1,2,1);
plot(1: m, output_train, 'r-*', 1: m, Train_sim, 'b-o', 'LineWidth', 1)
legend('真实值', '预测值(PSO)')
xlabel('预测样本')
ylabel('预测结果')
string = {'PSO-BP训练集预测结果对比'; ['准确率=' num2str(error1) '%']};
title(string)
xlim([1, m])
grid

subplot(1,2,2);
plot(1: m, output_train, 'r-*',1:m,Train_sim1,'g-^', 'LineWidth', 1)
legend('真实值','预测值(EGPSO)')
xlabel('预测样本')
ylabel('预测结果')
string1 = {'EGPSO-BP训练集预测结果对比'; ['准确率=' num2str(error3) '%']};
title(string1)
xlim([1, m])
grid

figure
subplot(1,2,1)
plot(1:n, output_test, 'r-*', 1: n, Test_sim, 'b-o', 'LineWidth', 1)
legend('真实值', '预测值(PSO)')
xlabel('预测样本')
ylabel('预测结果')
string3 = {'PSO-BP测试集预测结果对比'; ['准确率=' num2str(error2) '%']};
title(string3)
xlim([1, n])
grid

subplot(1,2,2)
plot(1:n, output_test, 'r-*',1:n,Test_sim1,'g-^', 'LineWidth', 1)
legend('真实值', '预测值(EGPSO)')
xlabel('预测样本')
ylabel('预测结果')
string4 = {'EGPSO-BP测试集预测结果对比'; ['准确率=' num2str(error4) '%']};
title(string4)
xlim([1, n])
grid

figure
subplot(1,2,1)
plot(1: length(bestfit),bestfit,'LineWidth', 1.5);
xlabel('粒子群迭代次数');
ylabel('适应度值');
xlim([1, length(bestfit)])
string = {'PSO-BP模型迭代误差变化'};
title(string)
grid on

subplot(1,2,2)
plot(1: length(bestfit1),bestfit1,'g-','LineWidth', 1.5);
xlabel('粒子群迭代次数');
ylabel('适应度值');
xlim([1, length(bestfit1)])
string = {'EGPSO-BP模型迭代误差变化'};
title(string)
grid on


toc