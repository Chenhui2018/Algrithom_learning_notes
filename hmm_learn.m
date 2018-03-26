%http
%https://cn.mathworks.com/help/stats/hidden-markov-models-hmm.html?requestedDomain=true

%状态转移矩阵
TRANS = [.9 .1; .05 .95];

%发射矩阵
EMIS = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6;...
7/12, 1/12, 1/12, 1/12, 1/12, 1/12];

% 建立1000个观测值，1000长度的观测链
% 创建了HMM（一种统计模型）
% seq是观测链， states是隐状态链
[seq,states] = hmmgenerate(1000,TRANS,EMIS);

% 假如已知观测系列，估计隐状态
likelystates = hmmviterbi(seq, TRANS, EMIS);
% 衡量估计的准确性
result = sum(states==likelystates)/1000;

% 假如已知观测序列和隐状态，求状态转移矩阵和发射矩阵
[TRANS_EST, EMIS_EST] = hmmestimate(seq, states);

% 假如仅知道观测序列，求状态转移矩阵和发射矩阵
% 状态转移矩阵和发射矩阵的初始化（估计）
TRANS_GUESS = [.85 .15; .1 .9];
EMIS_GUESS = [.17 .16 .17 .16 .17 .17;.6 .08 .08 .08 .08 08];
[TRANS_EST2, EMIS_EST2] = hmmtrain(seq, TRANS_GUESS, EMIS_GUESS);
% 设置训练的最大迭代次数
maxiter = 200;
hmmtrain(seq,TRANS_GUESS,EMIS_GUESS,'maxiterations',maxiter)
% 设置训练的最小误差
tol = 0.01;
hmmtrain(seq, TRANS_GUESS, EMIS_GUESS, 'tolerance', tol)


% PSTATES(i,j) 条件概率；状态i下，观测j发生的概率
% PSTATES KxN，K是状态数，N是观测类型总数
% 已知观测序列，求后验概率 
PSTATES = hmmdecode(seq,TRANS,EMIS);

% 返回观测序列seq的概率，取log返回[PSTATES,logpseq] = hmmdecode(seq,TRANS,EMIS);
% 为了防止出现取log前出现极其小的值，所以处理一下状态转移矩阵和发射矩阵% p=[p1,p2,...]，和为1
% 说明详见网址
TRANS_HAT = [0 p; zeros(size(TRANS,1),1) TRANS];
EMIS_HAT = [zeros(1,size(EMIS,2)); EMIS];


