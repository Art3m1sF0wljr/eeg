function X = solve_inverse_L1L1(B, A, L_t, lambda_t, rho, max_iter, tol)
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

    for iter = 1:max_iter
        X_prev = X;

        % --- Update X (least-squares) ---
        rhs = A' * (B - Z1 - U1) + (Z2 + U2) * L_t;
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

        % --- Check convergence ---
        primal_residual = norm(B - A * X - Z1, 'fro') + norm(X * L_t' - Z2, 'fro');
        dual_residual = rho * (norm(A' * (Z1 - Z1_prev), 'fro') + norm((Z2 - Z2_prev) * L_t, 'fro'));

        if primal_residual < tol && dual_residual < tol
            fprintf('Converged at iteration %d\n', iter);
            break;
        end
    end
end