function X = solve_inverse_L1_L1_v0(B, A, L_t, lambda_t, rho, max_iter, tol);
%min​{∥B−AX∥1​+λt​∥XLtT​∥1​}
%L(X,Z1​,Z2​,U1​,U2​)=∥Z1​∥1​+λt​∥Z2​∥1​+2ρ​(∥Z1​−B+AX+U1​∥22​+∥Z2​−XLtT​+U2​∥22​)
    % Inputs:
    %   B: EEG data (N_channels x T_time)
    %   A: Lead field matrix (N_channels x M_sources)
    %   L_t: Temporal Laplacian (T x T)
    %   lambda_t: Temporal regularization strength
    %   rho: ADMM penalty parameter (default: 1.0)
    %   max_iter: Maximum iterations (default: 100)
    %   tol: Convergence tolerance (default: 1e-6)
    %   verbose: Print progress (true/false, default: true)
    %
    % Output:
    %   X: Reconstructed sources (M_sources x T_time)

    verbose = true;

    [N, T] = size(B);
    M = size(A, 2);

    % Initialize variables
    X = zeros(M, T);
    Z1 = zeros(N, T);  % Auxiliary variable for data term
    Z2 = zeros(M, T);  % Auxiliary variable for temporal term
    U1 = zeros(N, T);  % Dual variable for data term
    U2 = zeros(M, T);  % Dual variable for temporal term

    % Precompute matrices for X-update
    AtA = A' * A;
    LtLt = L_t * L_t';
    I = eye(M);

    if verbose
        fprintf('ADMM optimization for L1-L1 problem\n');
        fprintf('===================================\n');
        fprintf('N = %d (channels), M = %d (sources), T = %d (time points)\n', N, M, T);
        fprintf('lambda_t = %.3f, rho = %.3f, max_iter = %d, tol = %.1e\n', lambda_t, rho, max_iter, tol);
        fprintf('-----------------------------------\n');
        fprintf('Iter\tPrimal Res\tDual Res\tObjective\n');
        fprintf('-----------------------------------\n');
    end

    for iter = 1:max_iter
        X_prev = X;
        Z1_prev = Z1;
        Z2_prev = Z2;

        % --- Update X (least-squares) ---
        residual_data = B - Z1_prev - U1;
        residual_temp = Z2_prev + U2;
        rhs = A' * residual_data + residual_temp * L_t;
        X = (AtA + rho * I) \ rhs;  % Solve using Cholesky or CG for large M

        % --- Update Z1 (data term, L1 proximal) ---
        residual = B - A * X - U1;
        Z1 = sign(residual) .* max(abs(residual) - 1/rho, 0);

        % --- Update Z2 (temporal term, L1 proximal) ---
        temporal_residual = X * L_t' - U2;
        Z2 = sign(temporal_residual) .* max(abs(temporal_residual) - lambda_t/rho, 0);

        % --- Update dual variables ---
        U1 = U1 + (B - A * X - Z1);
        U2 = U2 + (X * L_t' - Z2);

        % --- Compute residuals and objective ---
        primal_res = norm(B - A * X - Z1, 'fro') + norm(X * L_t' - Z2, 'fro');
        dual_res = rho * (norm(A' * (Z1 - Z1_prev), 'fro') + norm((Z2 - Z2_prev) * L_t, 'fro'));
		if primal_res > 10 * dual_res
			rho = rho * 1.1;   % Increase penalty if primal res is too high
		elseif dual_res > 10 * primal_res
			rho = rho / 1.1;   % Decrease penalty if dual res is too high
		end
        
        % Objective: ||B - AX||_1 + lambda_t ||X L_t^T||_1
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