function [X, history] = ADMM(S, lambda, rho)
% Solves the following problem:
% minimize  trace(S*X) - log det X + lambda*||X||_1
% with variable X, where S is the empirical covariance of the data
% matrix D (training observations by features).
%
% The solution is returned in the matrix X.
%
% history is a structure that contains the objective value, the primal and
% dual residual norms, and the tolerances for the primal and dual residual
% norms at each iteration.
%
% rho is the augmented Lagrangian parameter.

t_start = tic;
QUIET    = 1; % 1 for print, 0 for no print
MAX_ITER = 500;
ABSTOL   = 1e-4;
RELTOL   = 1e-2;
n = size(S,1);

% initilization
X = zeros(n);
Z = zeros(n);
U = zeros(n);

if ~QUIET
    fprintf('%3s\t%10s\t%10s\t%10s\t%10s\t%10s\n', 'iter', ...
        'r norm', 'eps pri', 's norm', 'eps dual', 'objective');
end

for k = 1:MAX_ITER

    % X-update
    [Q,L] = eig(rho*(Z - U) - S);
    es = diag(L);
    xi = (es + sqrt(es.^2 + 4*rho))./(2*rho);
    %xi = min((-es + sqrt(es.^2 + 4*rho))./(2*rho),1);
    X = Q*diag(xi)*Q';

    % Z-update
    Zold = Z;
    Z = shrinkage(X + U, lambda/rho);
    % U-update
    U = U + (X - Z);

    % diagnostics, reporting, termination checks

    history.objval(k)  = objective(S, X, Z, lambda);

    history.r_norm(k)  = norm(X - Z, 'fro'); % Frobenius norm
    history.s_norm(k)  = norm(-rho*(Z - Zold),'fro');

    history.eps_pri(k) = sqrt(n*n)*ABSTOL + RELTOL*max(norm(X,'fro'), norm(Z,'fro'));
    history.eps_dual(k)= sqrt(n*n)*ABSTOL + RELTOL*norm(rho*U,'fro');


    if ~QUIET
        fprintf('%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.4f\n', k, ...
            history.r_norm(k), history.eps_pri(k), ...
            history.s_norm(k), history.eps_dual(k), history.objval(k));
    end

    if (history.r_norm(k) < history.eps_pri(k) && ...
       history.s_norm(k) < history.eps_dual(k))
         break;
    end
end

if ~QUIET
    toc(t_start);
end
end

function obj = objective(S, X, Z, lambda)
obj = trace(S*X) - log(det(X)) + lambda*norm(Z(:), 1);
end

function y = shrinkage(a, kappa)
y = max(0, a-kappa) - max(0, -a-kappa);
end