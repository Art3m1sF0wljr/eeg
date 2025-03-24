function X = solve_inverse_L1_L1(B, A, L_t, lambda_t, rho, max_iter, tol)
    % Inputs:
    %   B: EEG data (N_channels x T_time)
    %   A: Lead field matrix (N_channels x M_sources)
    %   L_t: Temporal Laplacian (T x T)
    %   lambda_t: Temporal regularization strength
    %   rho: ADMM penalty parameter (default: 1.0)
    %   max_iter: Maximum iterations (default: 100)
    %   tol: Convergence tolerance (default: 1e-6)
    %   verbose_level: 0 (quiet), 1 (basic), 2 (detailed), 3 (debug)
    %
    % Output:
    %   X: Reconstructed sources (M_sources x T_time)

    % Set default parameters
    if nargin < 5 || isempty(rho), rho = 1.0; end
    if nargin < 6 || isempty(max_iter), max_iter = 100; end
    if nargin < 7 || isempty(tol), tol = 1e-6; end
    verbose_level = 3

    [N, T] = size(B);
    M = size(A, 2);

    % Initialize variables
    X = zeros(M, T);
    Z1 = zeros(N, T);
    Z2 = zeros(M, T);
    U1 = zeros(N, T);
    U2 = zeros(M, T);

    % Precompute matrices for X-update
    AtA = A' * A;
    I = eye(M);
    
    % Prepare LU factorization for faster solves
    [L, U, P] = lu(AtA + rho * I);

    % Verbosity header
    if verbose_level >= 1
        fprintf('\nADMM optimization for L1-L1 EEG inverse problem\n');
        fprintf('===============================================\n');
        fprintf('Dimensions: %d channels, %d sources, %d time points\n', N, M, T);
        fprintf('Parameters: lambda_t=%.2e, rho=%.2e, max_iter=%d, tol=%.1e\n', ...
                lambda_t, rho, max_iter, tol);
        
        if verbose_level >= 2
            fprintf('-----------------------------------------------\n');
            fprintf('Iter  Primal Res    Dual Res      Objective   Rho\n');
            fprintf('-----------------------------------------------\n');
        end
    end

    % Timing variables for verbosity
    start_time = tic;
    last_print_time = 0;
    print_interval = 2; % seconds between verbose updates

    for iter = 1:max_iter
        X_prev = X;
        Z1_prev = Z1;
        Z2_prev = Z2;

        % --- Update X ---
        residual_data = B - Z1_prev - U1;
        residual_temp = Z2_prev + U2;
        rhs = A' * residual_data + residual_temp * L_t;
        X = U \ (L \ (P * rhs));

        % --- Update Z1 ---
        residual = B - A * X - U1;
        Z1 = sign(residual) .* max(abs(residual) - 1/rho, 0);

        % --- Update Z2 ---
        temporal_residual = X * L_t' - U2;
        Z2 = sign(temporal_residual) .* max(abs(temporal_residual) - lambda_t/rho, 0);

        % --- Update dual variables ---
        U1 = U1 + (B - A * X - Z1);
        U2 = U2 + (X * L_t' - Z2);

        % --- Compute residuals and objective ---
        primal_res = norm(B - A * X - Z1, 'fro') + norm(X * L_t' - Z2, 'fro');
        dual_res = rho * (norm(A' * (Z1 - Z1_prev), 'fro') + norm((Z2 - Z2_prev) * L_t, 'fro'));
        obj = sum(abs(B - A * X), 'all') + lambda_t * sum(abs(X * L_t'), 'all');
        
        % --- Adaptive rho ---
        if primal_res > 10 * dual_res
            rho_new = rho * 2;
            rho_changed = true;
        elseif dual_res > 10 * primal_res
            rho_new = rho / 2;
            rho_changed = true;
        else
            rho_new = rho;
            rho_changed = false;
        end
        
        if rho_changed
            % Update factorization if rho changed
            rho = rho_new;
            [L, U, P] = lu(AtA + rho * I);
            
            if verbose_level >= 3
                fprintf('Rho updated to %.2e at iteration %d\n', rho, iter);
            end
        end

        % --- Verbose output ---
        current_time = toc(start_time);
        if verbose_level >= 2 && (current_time - last_print_time > print_interval || ...
                                 iter == 1 || iter == max_iter || ...
                                 (primal_res < tol && dual_res < tol))
            fprintf('%4d  %.2e  %.2e  %.2e  %.1e\n', ...
                   iter, primal_res, dual_res, obj, rho);
            last_print_time = current_time;
        end

        % --- Check convergence ---
        if primal_res < tol && dual_res < tol
            if verbose_level >= 1
                fprintf('-----------------------------------------------\n');
                fprintf('Converged in %d iterations (%.1f seconds)\n', iter, toc(start_time));
                fprintf('Final primal residual: %.2e\n', primal_res);
                fprintf('Final dual residual:   %.2e\n', dual_res);
                fprintf('Final objective:      %.2e\n', obj);
            end
            break;
        end
    end

    if iter == max_iter && verbose_level >= 1
        fprintf('-----------------------------------------------\n');
        fprintf('Maximum iterations reached (%.1f seconds)\n', toc(start_time));
        fprintf('Final primal residual: %.2e\n', primal_res);
        fprintf('Final dual residual:   %.2e\n', dual_res);
        fprintf('Final objective:      %.2e\n', obj);
    end
    
    if verbose_level >= 1
        fprintf('===============================================\n\n');
    end
end