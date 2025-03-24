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
    % Normalize data
    A = A / norm(A, 'fro');
    B = B / norm(B, 'fro');
    
    % Adaptive ρ parameters
    mu = 2;  % Threshold for residual balancing
    tau = 1;   % Update factor for ρ

    [N, T] = size(B);
    M = size(A, 2);

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
        Z1_prev = Z1;
        Z2_prev = Z2;

        % --- Update X using PCG ---
        rhs = A' * (B - Z1 - U1) + (Z2 + U2) * L_t;
        
        % Define the PCG function handle (returns a column vector)
        matvec = @(x) reshape(...
            A' * (A * reshape(x, M, T)) + rho * reshape(x, M, T), ...
            [], 1);  % Ensure column vector output
        
        % Solve with PCG
        X_vec = pcg(matvec, rhs(:), tol, 100);  % 100 inner iterations
        X = reshape(X_vec, M, T);

        % --- Update Z1 (L1 proximal operator) ---
        residual = B - A * X - U1;
        Z1 = sign(residual) .* max(abs(residual) - 1/rho, 0);

        % --- Update Z2 (L1 proximal operator) ---
        temporal_residual = X * L_t' - U2;
        Z2 = sign(temporal_residual) .* max(abs(temporal_residual) - lambda_t/rho, 0);

        % --- Update dual variables ---
        U1 = U1 + (B - A * X - Z1);
        U2 = U2 + (X * L_t' - Z2);

        % --- Compute residuals ---
        primal_res = norm(B - A * X - Z1, 'fro') + norm(X * L_t' - Z2, 'fro');
        dual_res = rho * (norm(A' * (Z1 - Z1_prev), 'fro') + norm((Z2 - Z2_prev) * L_t, 'fro'));
        
        % --- Adaptive ρ update ---
        if primal_res > mu * dual_res
            rho = rho * tau;
            U1 = U1 / tau;
            U2 = U2 / tau;
        elseif dual_res > mu * primal_res
            rho = rho / tau;
            U1 = U1 * tau;
            U2 = U2 * tau;
        end

        % Objective value
        obj = sum(abs(B - A * X), 'all') + lambda_t * sum(abs(X * L_t'), 'all');

        if verbose
            fprintf('%4d\t%.3e\t%.3e\t%.3e\t%.3e\n', iter, primal_res, dual_res, obj, rho);
        end

        % --- Check convergence ---
        if primal_res < tol && dual_res < tol
            if verbose
                fprintf('-----------------------------------\n');
                fprintf('Converged at iteration %d\n', iter);
            end
            break;
        end
    end

    if iter == max_iter && verbose
        fprintf('-----------------------------------\n');
        fprintf('Maximum iterations reached\n');
    end
end