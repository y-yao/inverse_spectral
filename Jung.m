% Algorithm from:
% A. Jung, G. Hannak and N. Goertz,
% "Graphical LASSO based Model Selection for Time Series,"
% IEEE Signal Processing Letters, vol. 22, no. 10, pp. 1781-1785, Oct. 2015.

clear;
data = 1; % 1 for synthetic; 2 for MRI_HC; 3 for MRI 4 for EEG

%% Choose Data File:
%filename = 'fMRI-TBI\FuncTimeSeries\NIH-101_20110331_ts.mat';
filename = 'HC_fMRI_2\NIH-218_20111215_fMRI_ROIts.mat';
%filename = 'rsEEG_AveCon\BM_MN_EO.mat';

%% Data preprocessing
if data ==1 % synthetic
    N = 200;
    p = 10;
    rng(138);
    cov = eye(p)*1;
    
    % Generate inverse covariance matrix
    % Uncomment one of the following two graph structures
    
    %(1) Star graph
    %cov(1,2:5) = .4;
    %cov(2:5,1) = .4;
    
    %(2) Loop graph
    cov(1,[2,4,5]) = .4;
    cov([2,4,5],1) = .4;
    cov(3,2) = .4; cov(2,3) = .4;
    cov(4,3) = .4; cov(3,4) = .4;
    
    % Moving-average model
    cov=inv(cov);
    noise = mvnrnd(zeros(1,p), cov, N); % N by p
    X = zeros(N-2,p);
    for i=3:N
        X(i-2,:) = 1/3*(noise(i,:)+noise(i-1,:)+noise(i-2,:));
    end
    N = N-2; % due to the above way of generating time series
    
elseif data == 2 % MRI healthy controls
    load(filename,'rvol')
    X=rvol.';
    N=200;p=86;
    
elseif data == 3 % MRI patients
    load(filename,'vol')
    X=vol.';
    N=200;p=86;
    
elseif data == 4 % EEG
    load(filename)
    X=1e11*AveEpoch(:,1:20:end).';
    [N,p] = size(X);
    
end

%% call GLASSO solver
F=4; % no. of frequency sampling points
lambda = 0.0000001; rho = .005;

[~,K]=gLASSO(X,F,.001,rho);

% Plotting
% gsupp_K=sum(abs(K),3);
% imagesc(gsupp_K)

% optional: normalize by diagonal entries
parfor f=1:F
    for i=1:p
        for j=1:p
            coh(i,j,f)=K(i,j,f)/sqrt(K(i,i,f)*K(j,j,f));
        end
    end
end
% Plotting
gsupp_coh=sum(abs(coh),3);
imagesc(gsupp_coh)
colorbar
