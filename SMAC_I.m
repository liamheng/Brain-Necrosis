function [out] = SMAC_I (data,imagesize,options)

% ======================================================================
% ======================================================================
% This algorithm is to solve the Fused Lasso Penalized Multi-Category
% Angle-Based Classification with Large Margin Unified Loss Function.
%
%
% ======================================================================
% Input:
% ======================================================================
% data is a Matlab structure which include train_x,train_y,tune_x,tune_y
%
% train_x,train_y are used to build the classfication model.
% tune_x,tune_y are used to tune the parameter and select the model.
% All the input imaging data are already vectorized, with each row is a
% subject and each colom is a pixel/voxel value.
%
% ======================================================================
%
% imagesize is a vector of length 2 or 3. The first element is the number
% of rows in the image, and the second element if the number of columns in
% the image. Only for 3-D image, the third element is the number of layers
% of the image.
%
% ======================================================================
%
% Options is a Matlab structure, which controls the operation of the
% algorithm.
%
% options.itmax: the max number of iterations allowed in the algorithm;
% options.rho: the augmented Lagragian parameter, by default is 1;
% options.l1flag: the flag variable indicating whether the intersecpt term
% will be penalized in the L1 penalty;
% options.tol: a threshold to control the convergence fo the algorithm;
% options.progress: the flag variable indicating whether the computation
% progress is displayed or not;
% options.lambda1: the vector of the possible choise of parameters of L1
% penalty;
% options.lambda2: the vector of the possible choise of parameters of total
% variation penalty;
%
%
% ======================================================================
% Output:
% ======================================================================
%
% out.b: the vectorized estimated coefficients, including the intercept
% out.tune_acc: the matrix of the tuning accuracy ;
% out.opt_lambda: the selected tuning parameter of lambda_1 and lambda_2
% in the tuning precedure;
%
% ======================================================================
%
% history.fvar: the loss function value in each iteration(for the
% optimized tuning parameter);
% history.tuning_time: tuning time for each pair of tuning parameter;
% history.conv: indicator of the convergence of the algorithm;
%
% ======================================================================
% ======================================================================




% ======================================================================
% ======================================================================
run('init.m')
% if a masking vector is predefined, set it as data.index
%
if  isfield(data,'index') ;
    %     if islogical(data.index) && isvector(data.index)
    index=data.index;
else
    %         warning('data.index must be a numerical vector with length equal to size of the image. All pixel/voxel are included')
    %         index = true(size(data.train_x,2),1);
    %     end
    % else
    index = true(size(data.train_x,2),1);
end


% ======================================================================
% ======================================================================
% checking imagesize input

d1 = imagesize(1);
if length(imagesize) > 3
    error('Error: Image size not supported, imagesize must be a vector of length 1,2 or 3')
elseif length(imagesize) == 3
    d3 = imagesize(3);
    d2 = imagesize(2);
elseif length(imagesize) == 2
    d3 = 1;
    d2 = imagesize(2);
elseif length(imagesize) == 1
    d3 = 1;
    d2 = 1;
end


if d1*d2*d3 ~=length(index)
    error('Error: Image size does not match the data size')
end


% ======================================================================
% ======================================================================
% setting options input

if isfield(options, 'l1flag')
    if isnumeric(options.l1flag)
        flag = options.l1flag;
    else
        %         warning('options.l1flag must be 0 or 1. A defaul value of 0 will be used');
        flag = 0;
    end
else
    flag = 0;
end

data.k = length(unique(data.train_y));
k = data.k; % total number of classes

if isfield(options, 'rho')
    if isnumeric(options.rho)&&(options.rho>0)
        rho = options.rho;
    else
        %         warning('options.rho must be a positive number. A default of 1 will be used here');
        rho = 1;
    end
else
    rho = 1;
end

if isfield(options, 'itmax')
    if isnumeric(options.itmax)
        itmax = options.itmax;
    else
        %         warning('options.itmax must be an integer. A default value of 500 will be used here')
        itmax = 500;
    end
else
    itmax = 500;
end

if isfield(options, 'tol')
    if isnumeric(options.tol)
        tol = options.tol;
    else
        %         warning('options.tol must be numeric. A default value of 10E-4 will be used here')
        tol = 10E-4;
    end
else
    tol = 10E-4;
end

if isfield(options,'lambda1');
    if isvector(options.lambda1) && isnumeric(options.lambda1)
        lambda1pool = options.lambda1;
    else
        %         warning('options.lambda1 must be a numeric vector. A default value of 8.^(-3:3) will be used here')
        lambda1pool = 8.^(-3:3);
    end
else
    lambda1pool = 8.^(-3:3);
end


if isfield(options,'lambda2');
    if isvector(options.lambda2) && isnumeric(options.lambda2)
        lambda2pool = options.lambda2;
    else
        %         warning('options.lambda1 must be a numeric vector. A default value of 8.^(-3:3) will be used here')
        lambda2pool = 8.^(-3:3);
    end
else
    lambda2pool = 8.^(-3:3);
end

% if isfield(options, 'progress')
%     if ~islogical(options.progress)
%         %         warning('options.progress must be true or false. A default of false will be used here')
%         options.progress = false;
%     end
% else
%     options.progress = false;
% end



% ======================================================================
% ======================================================================
% pre-caculate the necessary variables

p_org = length(index);
[R,~] = aug_mat(index,k);


[nobs,p] = size(data.train_x);


% the inverse part of updating b
A = W_gen(data.train_y,k)*X_cal(data.train_x,k);

At = A';
HL = At/(speye(nobs)+A*At);


% Pre-compute the selection matrix for updating v3
M = M_gen(d1,d2,d3,k,flag,index);


%% ======================================================================
%  start updating
%  ======================================================================


n_lambda1 = length(lambda1pool);
n_lambda2 = length(lambda2pool);


acc_tune = zeros(n_lambda1,n_lambda2);
tune_x = data.tune_x;
tune_y = data.tune_y;

load('y.mat');
b_opt = zeros((k-1)*(p+1),1);
acc_opt = 0;
for tune_i = 1:n_lambda1;
    for tune_j = 1:n_lambda2
        %======================================================================
        % initialize the parameters
        %======================================================================
        
        v1 = zeros(nobs,1);
        v2 = zeros((k-1)*(p_org+1),1);
        if d3>1
            v3 = zeros((4*p_org+1)*(k-1), 1);
        elseif d2>1
            v3 = zeros((3*p_org+1)*(k-1), 1);
        else
            v3 = zeros((2*p_org+1)*(k-1), 1);
        end
        
        u1 = v1;
        u2 = v2;
        u3 = v3;
        
        
        %======================================================================
        % tuning parameters
        %======================================================================
        
        lambda1 = lambda1pool(tune_i);
        lambda2 = lambda2pool(tune_j);
        
        [B,bccb] = B_gen(lambda1,lambda2,d1,d2,d3,k);
        
        b = zeros((k-1)*(p+1),1);
        if isfield(options, 'progress')
            if options.progress
                fprintf('Tuning Parameter: lambda1 = %2f and lambda2 = %2f\n',...
                    lambda1,lambda2);
            end
        end
        for t = 1:itmax
            tic;
            b_old = b;
            
            %======================================================================
            %======================================================================
            % update the y block
            %======================================================================
            %======================================================================
            
            
            %======================================================================
            % v1 updates component-wisely
            %======================================================================
            
            C_v1=A*b+u1;
            v1=(C_v1<-1).*(1+C_v1)+(ge(C_v1,-1)&le(C_v1,12)).*y(min(floor((abs(C_v1+1))*10^5)+1,1300001))...
                +(C_v1>12).*C_v1;
            %         v1(C_v1<-1,1)=1+C_v1(C_v1<-1);
            %         v1((ge(C_v1,-1)&le(C_v1,12)),1)=y(floor((C_v1(ge(C_v1,-1)&le(C_v1,12))+1)*10^5)+1);
            %         v1((C_v1>12))=C_v1(C_v1>12);
            %
            
            %======================================================================
            % v2 updates using FFT
            %======================================================================
            
            v2_b = (R*b+u2)+B'*(v3+u3);
            for ii = 1 : (k-1)
                v2 (1 + (ii-1)*(p_org+1),1) = v2_b(1 + (ii-1)*(p_org+1))/(1+lambda1^2);
                v2 ((2:p_org+1)+(ii-1)*(p_org+1),1) = invAb (bccb,...
                    v2_b((2:p_org+1)+(ii-1)*(p_org+1)),imagesize);
            end
            
            
            %======================================================================
            %======================================================================
            % update the x block
            %======================================================================
            %======================================================================
            
            %======================================================================
            % v3 updates using soft thresholding
            %======================================================================
            
            % For M_ii = 1, the entry is updated by soft thresholding.
            v31 = soft(B * v2-u3, 1/rho);
            
            % For M_ii = 0. the entry is updated by quadratic minimization.
            v32 = B * v2-u3;
            
            v3 = v32 + M * (v31 - v32);
            
            
            %======================================================================
            % b updates
            %======================================================================
            
            Kt = At*(v1-u1) + R'*(v2-u2);
            HRt = A*Kt;
            b = Kt - HL*HRt;
            
            
            %======================================================================
            %======================================================================
            % dual updates
            %======================================================================
            %======================================================================
            
            
            u1 = u1+(A*b-v1);
            u2 = u2+(R*b-v2);
            u3 = u3+(v3-B*v2);
            
            %======================================================================
            % check tolarence
            %======================================================================
            
            rel_change = norm(b-b_old)/norm(b_old);
            time_inner = toc;
            if isfield(options,'progress')
                if options.progress
                    timer_iner = toc;
                    fprintf('Iteration %4.0f/%4.0f\nTime elapsed = %2.2f\nRelative Change: %3.2f%%\n', t,itmax, time_inner,rel_change*100);
                end
            end
            if rel_change  < tol
                break
            end
        end
        [~,acc_new] = pred_ang(b,p,k,tune_x,tune_y);
        acc_tune(tune_i,tune_j) = acc_new;
        %======================================================================
        % Using the last solution as the initial for next update.
        %======================================================================
        if acc_new >= acc_opt
            b_opt = b;
            acc_opt = acc_new;
        end
        if isfield(options,'progress')
            if options.progress
                timer_iner = toc;
                %fprintf('Time elapsed = %2f\nTuning Accuracy: %3.2f%%\nBest Tuned Accuray: %3.2f%%\n', timer_iner,acc_new*100,acc_opt*100);
                fprintf('Tuning Accuracy: %3.2f%%\nBest Tuned Accuray: %3.2f%%\n',acc_new*100,acc_opt*100);
            end
        end
    end
end


out.tune_acc = acc_tune;

[max_row, max_col] = find(ismember(out.tune_acc, max(out.tune_acc(:))));
[~,loc] = max(max_row+max_col);
opt_row = max_row(loc);
opt_col = max_col(loc);
out.opt_lambda = [lambda1pool(opt_row);lambda2pool(opt_col)];
out.b = b_opt;
temp_coef = R*out.b;
if d2==1&&d3==1
    imgsize = [imagesize,1,1];
else
    imgsize = imagesize;
end
for i = 1:(k-1)
    eval( sprintf('out.coef_%d = reshape(temp_coef(((i-1)*(p_org+1)+2):i*(p_org+1)),imgsize);',i));
end

[out.pred_y,out.test_acc,out.conf] = pred_ang(out.b,size(data.test_x,2),data.k,data.test_x,data.test_y);



