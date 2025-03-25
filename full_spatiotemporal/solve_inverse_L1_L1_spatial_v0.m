function X = stabilized_solver(B, A, L_s, lambda_s, rho, max_iter, tol)
    [N, T] = size(B);
    M = size(A, 2);
    verbose = true;

    % Initialize with better scaling
    X = A' * B * 0.01;  % Warm start
    Z1 = zeros(N, T);
    Z2 = zeros(M, T);
    U1 = zeros(N, T);
    U2 = zeros(M, T);
    
    % Precompute with regularization
    AtA = A' * A + 1e-8*eye(M);  % Regularized
    LsTLs = L_s' * L_s + 1e-8*eye(M);
    if verbose
        fprintf('ADMM optimization for L1 with spatial Laplacian regularization\n');
        fprintf('============================================================\n');
        fprintf('N = %d (channels), M = %d (sources), T = %d (time points)\n', N, M, T);
        fprintf('lambda_s = %.3f, rho = %.3f, max_iter = %d, tol = %.1e\n', lambda_s, rho, max_iter, tol);
        fprintf('------------------------------------------------------------\n');
        fprintf('Iter\tPrimal Res\tDual Res\tObjective\tRho\n');
        fprintf('------------------------------------------------------------\n');
    end
    for iter = 1:max_iter
        % X-update with iterative refinement
        rhs = A'*(B-Z1-U1) + rho*L_s'*(Z2+U2);
        X_prev = X;
		Z1_prev=Z1;
		Z2_prev=Z2;
        for refine = 1:3  % Inner refinement
            X = (AtA + rho*LsTLs) \ rhs;
            if norm(X - X_prev,'fro') < 1e-6
                break;
            end
            X_prev = X;
        end
        
        % Relaxed Z-updates
        Z1 = soft_threshold(B - A*X - U1, 0.95/rho);
        Z2 = soft_threshold(L_s*X - U2, 0.95*lambda_s/rho);
        
        % Adaptive rho with safeguards
        primal_res = norm(B-A*X-Z1,'fro') + norm(L_s*X-Z2,'fro');
        dual_res = rho*(norm(A'*(Z1-Z1_prev),'fro') + norm(L_s'*(Z2-Z2_prev),'fro'));
        
        if primal_res > 5*dual_res
            rho = min(rho*1.5, 100);
        elseif dual_res > 5*primal_res
            rho = max(rho/1.5, 0.1);
        end
        % Objective: ||B - AX||_1 + lambda_s ||L_s X||_1
        obj = sum(abs(B - A * X), 'all') + lambda_s * sum(abs(L_s * X), 'all');

        if verbose
            fprintf('%4d\t%.3e\t%.3e\t%.3e\t%.3f\n', iter, primal_res, dual_res, obj, rho);
        end

        % --- Check convergence ---
        if primal_res < tol && dual_res < tol
            if verbose
                fprintf('------------------------------------------------------------\n');
                fprintf('Converged at iteration %d\n', iter);
            end
            break;
        end
        % Early termination check
        if iter > 20 && primal_res > 1e3*min_primal_res
            warning('Divergence detected - restarting with new parameters');
            rho = rho*2;
            continue;
        end
    end
end

function X = soft_threshold(X, threshold)
    X = sign(X) .* max(abs(X) - threshold, 0);
end