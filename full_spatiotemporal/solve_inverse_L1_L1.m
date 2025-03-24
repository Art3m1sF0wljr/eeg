function X = solve_inverse_L1_L1(B, A, L_t, lambda_t, rho, max_iter, tol)
    % Inputs:
    %   B: EEG data (N_channels x T_time)
    %   A: Lead field matrix (N_channels x M_sources)
    %   L_t: Temporal Laplacian (T x T)
    %   lambda_t: Temporal regularization strength
    %   rho: ADMM penalty parameter (default: 1.0)
    %   max_iter: Maximum iterations (default: 100)
    %   tol: Convergence tolerance (default: 1e-6)
    %
    % Output:
    %   X: Reconstructed sources (M_sources x T_time)

    verbose = true;
    % Normalize problem
    scale_A = norm(A, 'fro');
    A = A / scale_A;
    B = B / scale_A;
    
    % Adaptive ρ parameters
    mu = 5;  % Threshold for residual balancing
    tau = 1.5;   % Update factor for ρ

    [N, T] = size(B);
    M = size(A, 2);

	AtA = A' * A;
    I = eye(M);
    Lt = L_t';
	
    % Initialize variables
    X = zeros(M, T);
    Z1 = zeros(N, T);
    Z2 = zeros(M, T);
    U1 = zeros(N, T);
    U2 = zeros(M, T);

    if verbose
        fprintf('ADMM optimization for L1-L1 problem\n');
        fprintf('===================================\n');
        fprintf('N = %d (channels), M = %d (sources), T = %d (time points)\n', N, M, T);
        fprintf('lambda_t = %.3f, rho = %.3f, max_iter = %d, tol = %.1e\n', lambda_t, rho, max_iter, tol);
        fprintf('-----------------------------------\n');
        fprintf('Iter\tPrimal Res\tDual Res\tObjective\tRho\n');
        fprintf('-----------------------------------\n');
    end

    for iter = 1:max_iter
        X_prev = X;
        
        % --- X-update: Solve (A'*A + ρI)X = A'(B-Z1-U1) + (Z2+U2)*L_t ---
        RHS = A' * (B - Z1 - U1) + (Z2 + U2) * L_t;
        
        % Use Jacobi preconditioned PCG
        preconditioner = diag(diag(AtA) + rho;
        [X_vec, ~] = pcg(@(x) reshape(AtA * reshape(x, M, T) + rho * reshape(x, M, T), ...
                         RHS(:), 1e-6, 100, diag(1./preconditioner));
        X = reshape(X_vec, M, T);

        % --- Z-updates (L1 proximal operators) ---
        Z1 = soft_threshold(B - A * X - U1, 1/rho);
        Z2 = soft_threshold(X * Lt - U2, lambda_t/rho);

        % --- Dual updates ---
        U1 = U1 + (B - A * X - Z1);
        U2 = U2 + (X * Lt - Z2);

        % --- Residuals ---
        primal_res = norm(B - A * X - Z1, 'fro') + norm(X * Lt - Z2, 'fro');
        dual_res = rho * norm(A' * (Z1 - Z1_prev) + (Z2 - Z2_prev) * L_t, 'fro');

        % --- Adaptive rho update ---
        if primal_res > mu * dual_res
            rho = min(rho * tau, max_rho);
            U1 = U1 / tau;
            U2 = U2 / tau;
        elseif dual_res > mu * primal_res
            rho = max(rho / tau, min_rho);
            U1 = U1 * tau;
            U2 = U2 * tau;
        end

        fprintf('%4d\t%.2e\t%.2e\t%.2e\n', iter, primal_res, dual_res, rho);

        if max(primal_res, dual_res) < tol
            fprintf('Converged!\n');
            break;
        end
    end
end

function Y = soft_threshold(X, threshold)
    Y = sign(X) .* max(abs(X) - threshold, 0);
end