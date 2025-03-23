function J_reconstructed = ADMM_L1_minimization(B, A, L_s, L_t, lambda_s, lambda_t, rho, max_iter, tol)
    % ADMM_L1_minimization: Solves the L1-regularized inverse problem using ADMM.
    %
    % Inputs:
    %   B: Measured EEG data (Nch x T)
    %   A: Leadfield matrix (Nch x Nsources)
    %   L_s: Spatial Laplacian (Nsources x Nsources)
    %   L_t: Temporal Laplacian (T x T)
    %   lambda_s: Spatial regularization parameter
    %   lambda_t: Temporal regularization parameter
    %   rho: ADMM penalty parameter
    %   max_iter: Maximum number of iterations
    %   tol: Convergence tolerance
    %
    % Output:
    %   J_reconstructed: Reconstructed source activity (Nsources x T)

    % Dimensions
    [Nch, T] = size(B);
    Nsources = size(A, 2);

    % Initialize variables
    X = zeros(Nsources, T);
    Z1 = zeros(Nch, T);
    Z2 = zeros(Nsources, T);
    Z3 = zeros(Nsources, T);
    Y1 = zeros(Nch, T);
    Y2 = zeros(Nsources, T);
    Y3 = zeros(Nsources, T);

    % Precompute matrices
    ATA = A' * A;
    LsTLs = L_s' * L_s;
    LtLtT = L_t * L_t';

    % ADMM iterations
    for iter = 1:max_iter
        % Update X using an iterative solver (PCG)
        RHS = A' * (B - Z1 + Y1) + L_s' * (Z2 - Y2) + (Z3 - Y3) * L_t';
        X_vec = pcg(@(x) apply_system_matrix(x, ATA, LsTLs, LtLtT, Nsources, T), RHS(:), tol, 1000);
        X = reshape(X_vec, [Nsources, T]);

        % Update Z variables using soft-thresholding
        Z1_prev = Z1; Z2_prev = Z2; Z3_prev = Z3;
        Z1 = soft_threshold(B - A * X + Y1, 1 / rho);
        Z2 = soft_threshold(L_s * X + Y2, lambda_s / rho);
        Z3 = soft_threshold(X * L_t' + Y3, lambda_t / rho);

        % Update dual variables
        Y1 = Y1 + (B - A * X - Z1);
        Y2 = Y2 + (L_s * X - Z2);
        Y3 = Y3 + (X * L_t' - Z3);

        % Check convergence
        primal_residual = norm(B - A * X - Z1, 'fro') + norm(L_s * X - Z2, 'fro') + norm(X * L_t' - Z3, 'fro');
        dual_residual = rho * (norm(A' * (Z1 - Z1_prev), 'fro') + norm(L_s' * (Z2 - Z2_prev), 'fro') + norm((Z3 - Z3_prev) * L_t, 'fro'));

        if primal_residual < tol && dual_residual < tol
            fprintf('Converged at iteration %d\n', iter);
            break;
        end
    end

    % Output reconstructed source activity
    J_reconstructed = X;
end

% Function to apply system matrix M to a vector x
function y = apply_system_matrix(x, ATA, LsTLs, LtLtT, Nsources, T)
    X = reshape(x, [Nsources, T]);
    Y = (ATA + LsTLs) * X + X * LtLtT;
    y = Y(:);
end

% Soft-thresholding function
function y = soft_threshold(x, tau)
    y = sign(x) .* max(abs(x) - tau, 0);
end
