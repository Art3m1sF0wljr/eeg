function J_reconstructed = ADMM_L1_minimization_v0(B, A, L_s, L_t, lambda_s, lambda_t, rho, max_iter, tol)
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
    X = zeros(Nsources, T);  % Source activity
    Z1 = zeros(Nch, T);      % Auxiliary variable for data fidelity
    Z2 = zeros(Nsources, T); % Auxiliary variable for spatial regularization
    Z3 = zeros(Nsources, T); % Auxiliary variable for temporal regularization
    U1 = zeros(Nch, T);      % Dual variable for data fidelity
    U2 = zeros(Nsources, T); % Dual variable for spatial regularization
    U3 = zeros(Nsources, T); % Dual variable for temporal regularization

    % Precompute matrices for X update
    ATA = A' * A;  % Size: Nsources x Nsources
    LsTLs = L_s' * L_s;  % Size: Nsources x Nsources
    LtLtT = L_t * L_t';  % Size: T x T

    % ADMM iterations
    for iter = 1:max_iter
        % Update X (vectorized)
        RHS = A' * (Z1 + B - U1);  % Size: Nsources x T

        % Spatial regularization term
        spatial_term = L_s' * (Z2 - U2);  % Size: Nsources x T

        % Temporal regularization term
        temporal_term = (Z3 - U3) * L_t';  % Size: Nsources x T

        % Combine terms
        RHS = RHS + spatial_term + temporal_term;

        % Solve M * X_vec = RHS using an iterative solver (e.g., PCG)
        X_vec = pcg(@(x) apply_system_matrix(x, ATA, LsTLs, LtLtT, Nsources, T), RHS(:), tol, 1000);

        % Reshape X back to matrix form
        X = reshape(X_vec, [Nsources, T]);

        % Update Z1 (soft-thresholding for data fidelity)
        residual1 = B - A * X + U1;
        Z1 = soft_threshold(residual1, 1 / rho);

        % Update Z2 (soft-thresholding for spatial regularization)
        residual2 = L_s * X + U2;
        Z2 = soft_threshold(residual2, lambda_s / rho);

        % Update Z3 (soft-thresholding for temporal regularization)
        residual3 = X * L_t' + U3;
        Z3 = soft_threshold(residual3, lambda_t / rho);

        % Update dual variables
        U1 = U1 + (B - A * X - Z1);
        U2 = U2 + (L_s * X - Z2);
        U3 = U3 + (X * L_t' - Z3);

        % Check convergence
        primal_residual = norm(B - A * X - Z1, 'fro') + ...
                          norm(L_s * X - Z2, 'fro') + ...
                          norm(X * L_t' - Z3, 'fro');
        dual_residual = rho * (norm(A' * (Z1 - Z1_prev), 'fro') + ...
                              norm(L_s' * (Z2 - Z2_prev), 'fro') + ...
                              norm((Z3 - Z3_prev) * L_t, 'fro'));

        if primal_residual < tol && dual_residual < tol
            fprintf('Converged at iteration %d\n', iter);
            break;
        end

        % Store previous Z values for dual residual calculation
        Z1_prev = Z1;
        Z2_prev = Z2;
        Z3_prev = Z3;
    end

    % Output the reconstructed source activity
    J_reconstructed = X;
end

% Function to apply the system matrix M to a vector x
function y = apply_system_matrix(x, ATA, LsTLs, LtLtT, Nsources, T)
    % Reshape x into a matrix
    X = reshape(x, [Nsources, T]);

    % Compute M * x using the properties of the Kronecker product
    Y = (ATA + LsTLs) * X + X * LtLtT;

    % Reshape back into a vector
    y = Y(:);
end

% Soft-thresholding function
function y = soft_threshold(x, tau)
    y = sign(x) .* max(abs(x) - tau, 0);
end