function X = solve_inverse_L1_L1_spatial_v0(B, A, L_s, lambda_s, rho, max_iter, tol)
% Solves: min{∥B−AX∥₁ + λₛ∥LₛX∥₁}
% Using ADMM with the augmented Lagrangian:
% L(X,Z1,Z2,U1,U2) = ∥Z1∥₁ + λₛ∥Z2∥₁ + (ρ/2)(∥Z1−B+AX+U1∥₂² + ∥Z2−LₛX+U2∥₂²)
%
% Inputs:
%   B: EEG data (N_channels × T_time)
%   A: Lead field matrix (N_channels × M_sources)
%   L_s: Spatial Laplacian matrix (M_sources × M_sources)
%   lambda_s: Spatial regularization strength
%   rho: ADMM penalty parameter (default: 1.0)
%   max_iter: Maximum iterations (default: 100)
%   tol: Convergence tolerance (default: 1e-6)
%
% Output:
%   X: Reconstructed sources (M_sources × T_time)

    % Set defaults for optional parameters

    
    verbose = true;

    [N, T] = size(B);
    M = size(A, 2);

    % Initialize variables
    X = zeros(M, T);
    Z1 = zeros(N, T);  % Auxiliary variable for data term
    Z2 = zeros(M, T);  % Auxiliary variable for spatial term
    U1 = zeros(N, T);  % Dual variable for data term
    U2 = zeros(M, T);  % Dual variable for spatial term

    % Precompute matrices for X-update
    AtA = A' * A;
    LsTLs = L_s' * L_s;
    I = eye(M);

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
        X_prev = X;
        Z1_prev = Z1;
        Z2_prev = Z2;

        % --- Update X (least-squares) ---
        % Solve: (A'A + ρ Lₛ'Lₛ)X = A'(B - Z1 - U1) + ρ Lₛ'(Z2 + U2)
        rhs = A' * (B - Z1_prev - U1) + rho * L_s' * (Z2_prev + U2);
        X = (AtA + rho * LsTLs) \ rhs;

        % --- Update Z1 (data term, L1 proximal) ---
        residual = B - A * X - U1;
        Z1 = sign(residual) .* max(abs(residual) - 1/rho, 0);

        % --- Update Z2 (spatial term, L1 proximal) ---
        spatial_residual = L_s * X - U2;
        Z2 = sign(spatial_residual) .* max(abs(spatial_residual) - lambda_s/rho, 0);

        % --- Update dual variables ---
        U1 = U1 + (B - A * X - Z1);
        U2 = U2 + (L_s * X - Z2);

        % --- Compute residuals and objective ---
        primal_res = norm(B - A * X - Z1, 'fro') + norm(L_s * X - Z2, 'fro');
        dual_res = rho * (norm(A' * (Z1 - Z1_prev), 'fro') + norm(L_s' * (Z2 - Z2_prev), 'fro'));
        
        % Adaptive rho adjustment
        if primal_res > 10 * dual_res
            rho = rho * 1.1;
        elseif dual_res > 10 * primal_res
            rho = rho / 1.1;
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
    end

    if iter == max_iter && verbose
        fprintf('------------------------------------------------------------\n');
        fprintf('Maximum iterations reached\n');
    end
end