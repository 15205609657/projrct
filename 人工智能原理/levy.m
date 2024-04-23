%% levy: 莱维飞行机制
% 输入
% pop：种群个体 
% VarMin：下限  
% VarMax：上限
% 输出
% newpop： 变异后的种群个体
function newpop=levy(pop,VarMin,VarMax)
[N,D]=size(pop);
beta=1.5;    % 通常取值为1.5
sigma_u=(gamma(1+beta)*sin(pi*beta/2)/(beta*gamma((1+beta)/2)*2^((beta-1)/2)))^(1/beta);
sigma_v=1;  
u=normrnd(0,sigma_u,N,D);    % 产生正态分布的随机数
v=normrnd(0,sigma_v,N,D);
step=u./(abs(v).^(1/beta));    % 步长
pop=pop+1.*step ;
% 处理边界
pop(find(pop>VarMax))=VarMax; 
pop(find(pop<VarMin))=VarMin;
newpop=pop;
end