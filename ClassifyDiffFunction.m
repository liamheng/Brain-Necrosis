function ClassifyDiffFunction(RoundNumber,mode,Windowsize)
% dbstop if error
% RoundNumber=1;
% moduality='Flair';
% Windowsize=6;
 % add full path of the package
moduality=[{'Flair'},{'T1'},{'T1c'},{'T2'}];
rng(RoundNumber)
load(['Diff_',moduality{mode},'_S',num2str(Windowsize)]);
run('init.m')
NegativeTe=randsample(find(ydata==1),3);
NegativeTr=setdiff(find(ydata==1),NegativeTe);    
PositiveTe=randsample(find(ydata==2),7);
PositiveTr=setdiff(find(ydata==2),PositiveTe);


%% Classification
Y=ydata([NegativeTr;PositiveTr]);
Ytest=ydata([NegativeTe;PositiveTe]);
ytraining=xdata([NegativeTr;PositiveTr],:,:);
ytesting=xdata([NegativeTe;PositiveTe],:,:);

num_shuffles=20; num_folds=2;
options.itmax=1000;  % Max iteration, default is 500.
options.rho=1; % Lagrangian parameter, default is 1.
options.l1flag =0; % Indicator of penalizing intercept. Default is 0.
options.tol=5E-4; % Tolerance for the convergence of the algorithm. Default is 10E-4.
options.progress = false; % Display the tuning progress. Default is false.
options.lambda1 = 2.^[-5:2:5]; % Penalty level for L1. Default is 8.^(-3:3);
options.lambda2 = 2.^[-5:2:5]; % Penalty level for TV. Default is 8.^(-3:3);
disp('Training start')
for k=1:size(ytraining,3)   % number of features
    x=ytraining(:,:,k);
%     x=reshape(x,[size(x,1),100,size(x,2)/100]);
    xtest=ytesting(:,:,k);
%     xtest=reshape(xtest,[size(xtest,1),100,size(xtest,2)/100]);
% x=reshape(y,[510 1100]);
for j = 1:num_shuffles    
    indices = crossvalind('Kfold',Y,num_folds);
data.train_x = x(indices==1,:,:); % train images, n by p
data.train_y = Y(indices==1); % train labels, n by 1
data.tune_x = x(indices==2,:,:); % tune images, n by p
data.tune_y = Y(indices==2); % tune labels, n by 1
% data.test_x = x(indices==3,:); % test images, n by p
% data.test_y = Y(indices==3); % test labels, n by 1
data.test_x = xtest; % test images, n by p
data.test_y = Ytest; % test labels, n by 1
data.index = ones(size(x,2),1); % logical p by 1
imagesize = 100;  % image size 1 by 3 or 2


% [out] = FLAC (data,imagesize,options); % run the program
% [out1] = SMAC_I (data,imagesize,options); % run the classification with TV-I penalty
[out2] = SMAC_II (data,imagesize,options); % run the classification with TV-II penalty

% OutPutMatrix{j}=out;
% accu1(j)=out1.test_acc;
% accu2(j)=out2.test_acc;
% [AUC1(j)] = ROC_analysis(data.test_y,data.test_x*out1.coef_1(:),0,'SMAC-I'); 
[AUC2(j)] = ROC_analysis(data.test_y,data.test_x*out2.coef_1(:),0,'SMAC-II');

end
fea_auc2(k)=median(AUC2)

save(['Diff_',moduality{mode},'_S',num2str(Windowsize),'_R',num2str(RoundNumber)],'fea_auc2')
end
