% Algorithm from:
% D. Schneider-Luftman and A. T. Walden, 
% "Partial Coherence Estimation via Spectral Matrix Shrinkage under Quadratic Loss," 
% IEEE Transactions on Signal Processing, vol. 64, no. 22, pp. 5767-5777, Nov.15, 15 2016.

clear;
data = 1; % 1 for synthetic; 2 for MRI_HC; 3 for MRI 4 for EEG
method = 2; % 1=inverse, 2=HS, 3=QL

%% Choose Data File:
filename = 'fMRI-TBI\FuncTimeSeries\NIH-101_20100426_ts.mat';
%filename = 'HC_fMRI_2\NIH-129_20100217_fMRI_ROIts.mat';
%filename = 'rsEEG_AveCon\BM_MN_EO.mat';

%% Data preprocessing
if data ==1 % synthetic
    N = 500;
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
    cov=inv(cov);
    
    % Moving-average model
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
    X=AveEpoch(:,1:20:end).';
    [N,p] = size(X);
end

%% 
F=4; % no. of frequencies
K=p+10; % See Sec III.B in paper
h =@(k,t) sqrt(2/(N+1))*sin((k+1)*pi*(t+1)/(N+1));
J=zeros(p,K,F);
for f=1:F
    for k=1:K
        for t=1:N
            J(:,k,f) = J(:,k,f)+h(k,t)*X(t,:).'*exp(-1i*2*pi*t*(f-1)/F);
        end
    end
end

S=zeros(p,p,F);
for f=1:F
    S(:,:,f) = (1/K)*J(:,:,f)*J(:,:,f)';
end


if method == 1 %% inverse
    S_inv=zeros(p,p,F);coh=zeros(p,p,f);
    parfor f=1:F
        S_inv(:,:,f) = inv(S(:,:,f));

    end
    parfor f=1:F
        for i=1:p
            for j=1:p
                coh(i,j,f)=S_inv(i,j,f)/sqrt(S_inv(i,i,f)*S_inv(j,j,f));
            end
        end
    end

elseif method ==2 %% Hilbert-Schmidt
    fprintf('==========HS==========\n')
    S_HS = zeros(p,p,F);
    S_HS_inv = zeros(p,p,F);coh_HS=zeros(p,p,f);
    parfor f=1:F
        tr_S = trace(S(:,:,f));
        tr_S2 = trace(S(:,:,f)^2) - tr_S^2/K;
        eta_0 = tr_S/p;
        rho_0 = real((1-K/p+K*tr_S2/tr_S^2)^(-1));
        S_HS(:,:,f) = (1-rho_0)*S(:,:,f) + rho_0*eta_0*eye(p);
        S_HS_inv(:,:,f) = inv(S_HS(:,:,f));

    end
    parfor f=1:F
        for i=1:p
            for j=1:p
                coh_HS(i,j,f)=S_HS_inv(i,j,f)/sqrt(S_HS_inv(i,i,f)*S_HS_inv(j,j,f));
            end
        end
    end
    
elseif method ==3 %% Quadratic Loss
    fprintf('==========QL==========\n')
    S_QL = zeros(p,p,F);
    S_QL_inv = zeros(p,p,F);coh_QL=zeros(p,p,f);
    for f=1:F
        tr_S_inv = trace(inv(S(:,:,f)));
        tr_S2_inv = trace(inv(S(:,:,f))^2);
        eta_0 = tr_S_inv/tr_S2_inv;
        rho_0 = real((1+K/p-K*tr_S_inv^2/p^2/tr_S2_inv)^(-1));
        S_QL(:,:,f) = (1-rho_0)*S(:,:,f) + rho_0*eta_0*eye(p);
        S_QL_inv(:,:,f) = inv(S_QL(:,:,f));

    end
    parfor f=1:F
        for i=1:p
            for j=1:p
                coh_QL(i,j,f)=S_QL_inv(i,j,f)/sqrt(S_QL_inv(i,i,f)*S_QL_inv(j,j,f));
            end
        end
    end
end

colormap jet;
if method == 1
    gsupp_K = sum(abs(coh),3);
elseif method == 2
    gsupp_K = sum(abs(coh_HS),3);
elseif method == 3
    gsupp_K = sum(abs(coh_QL),3);
end
imagesc(gsupp_K)
colorbar