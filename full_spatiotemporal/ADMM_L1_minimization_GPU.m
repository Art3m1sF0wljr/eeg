function X_reconstructed = ADMM_L1_minimization_GPU(B, A, L_s, L_t, lambda_s, lambda_t, rho, max_iter, tol, verbose)
    % ADMM_L1_minimization_GPU: Solves the L1-regularized inverse problem using ADMM with GPU acceleration.
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
    %   verbose: Verbosity level (0: no output, 1: print residuals, 2: print all variables)
    %
    % Output:
    %   X_reconstructed: Reconstructed source activity (Nsources x T)

    % Transfer data to GPU
    A = gpuArray(A);
    L_s = gpuArray(L_s);
    L_t = gpuArray(L_t);
    B = gpuArray(B);

    % Dimensions
    [Nch, T] = size(B);
    Nsources = size(A, 2);

    % Initialize variables on GPU
    X = gpuArray.zeros(Nsources, T);  % Source activity
    Z1 = gpuArray.zeros(Nch, T);      % Auxiliary variable for data fidelity
    Z2 = gpuArray.zeros(Nsources, T); % Auxiliary variable for spatial regularization
    Z3 = gpuArray.zeros(Nsources, T); % Auxiliary variable for temporal regularization
    U1 = gpuArray.zeros(Nch, T);      % Dual variable for data fidelity
    U2 = gpuArray.zeros(Nsources, T); % Dual variable for spatial regularization
    U3 = gpuArray.zeros(Nsources, T); % Dual variable for temporal regularization

    % Initialize previous Z values for dual residual calculation
    Z1_prev = Z1;
    Z2_prev = Z2;
    Z3_prev = Z3;

    % Precompute matrices for X update
    ATA = A' * A;  % Size: Nsources x Nsources
    LsTLs = L_s' * L_s;  % Size: Nsources x Nsources
    LtLtT = L_t * L_t';  % Size: T x T

    % Preconditioner setup (diagonal preconditioner)
    % Compute the diagonal of the system matrix implicitly
    diag_ATA_LsTLs = diag(ATA + LsTLs); % Diagonal of ATA + LsTLs (Nsources x 1)
    diag_LtLtT = diag(LtLtT); % Diagonal of LtLtT (T x 1)
    
    % Expand diag_ATA_LsTLs to match the size of diag_kron
    diag_ATA_LsTLs_expanded = repmat(diag_ATA_LsTLs, [T, 1]); % Repeat Nsources diagonal T times
    
    % Compute the diagonal of the Kronecker product term
    diag_kron = repelem(diag_LtLtT, Nsources); % Repeat each element of diag_LtLtT Nsources times
    
    % Combine the diagonals
    M_diag = diag_ATA_LsTLs_expanded + diag_kron; % Diagonal of the system matrix
    preconditioner = @(x) x ./ M_diag; % Diagonal preconditioner

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

        % Solve M * X_vec = RHS using custom GPU-accelerated PCG
        max_iter_pcg = 5000; % Increase the maximum number of iterations for PCG
        [X_vec, flag, relres, pcg_iter] = pcg_gpu(@(x) apply_system_matrix_implicit_gpu(x, ATA, LsTLs, LtLtT, Nsources, T), ...
                                             RHS(:), tol, max_iter_pcg, preconditioner);

        % Check PCG convergence
        if flag ~= 0 && verbose >= 1
            warning('PCG did not converge to the desired tolerance. Flag: %d, Relative residual: %e, Iterations: %d', flag, relres, pcg_iter);
        end

        % Reshape X back to matrix form
        X = reshape(X_vec, [Nsources, T]);

        % Update Z1 (soft-thresholding for data fidelity)
        residual1 = B - A * X + U1;
        Z1 = soft_threshold_gpu(residual1, 1 / rho);

        % Update Z2 (soft-thresholding for spatial regularization)
        residual2 = L_s * X + U2;
        Z2 = soft_threshold_gpu(residual2, lambda_s / rho);

        % Update Z3 (soft-thresholding for temporal regularization)
        residual3 = X * L_t' + U3;
        Z3 = soft_threshold_gpu(residual3, lambda_t / rho);

        % Update dual variables
        U1 = U1 + (B - A * X - Z1);
        U2 = U2 + (L_s * X - Z2);
        U3 = U3 + (X * L_t' - Z3);

        % Compute primal and dual residuals
        primal_residual = norm(B - A * X - Z1, 'fro') + ...
                          norm(L_s * X - Z2, 'fro') + ...
                          norm(X * L_t' - Z3, 'fro');
        dual_residual = rho * (norm(A' * (Z1 - Z1_prev), 'fro') + ...
                              norm(L_s' * (Z2 - Z2_prev), 'fro') + ...
                              norm((Z3 - Z3_prev) * L_t, 'fro'));

        % Verbosity: Print progress
        if verbose >= 1
            fprintf('Iteration %d:\n', iter);
            fprintf('  Primal Residual: %e\n', gather(primal_residual)); % Gather to print on CPU
            fprintf('  Dual Residual: %e\n', gather(dual_residual)); % Gather to print on CPU
            if verbose >= 2
                fprintf('  X: %s\n', mat2str(gather(X(1:min(3, Nsources), 1:min(3, T))))); % Print a small subset of X
                fprintf('  Z1: %s\n', mat2str(gather(Z1(1:min(3, Nch), 1:min(3, T))))); % Print a small subset of Z1
                fprintf('  Z2: %s\n', mat2str(gather(Z2(1:min(3, Nsources), 1:min(3, T))))); % Print a small subset of Z2
                fprintf('  Z3: %s\n', mat2str(gather(Z3(1:min(3, Nsources), 1:min(3, T))))); % Print a small subset of Z3
                fprintf('  U1: %s\n', mat2str(gather(U1(1:min(3, Nch), 1:min(3, T))))); % Print a small subset of U1
                fprintf('  U2: %s\n', mat2str(gather(U2(1:min(3, Nsources), 1:min(3, T))))); % Print a small subset of U2
                fprintf('  U3: %s\n', mat2str(gather(U3(1:min(3, Nsources), 1:min(3, T))))); % Print a small subset of U3
            end
        end

        % Check convergence
        if primal_residual < tol && dual_residual < tol
            if verbose >= 1
                fprintf('Converged at iteration %d\n', iter);
            end
            break;
        end

        % Store previous Z values for dual residual calculation
        Z1_prev = Z1;
        Z2_prev = Z2;
        Z3_prev = Z3;
    end

    % Output the reconstructed source activity (transfer back to CPU if needed)
    X_reconstructed = gather(X);
end
function y = apply_system_matrix_implicit_gpu(x, ATA, LsTLs, LtLtT, Nsources, T)
    % apply_system_matrix_implicit_gpu: Computes M * x implicitly on the GPU.
    %
    % Inputs:
    %   x: Input vector (gpuArray)
    %   ATA: A' * A (gpuArray, Nsources x Nsources)
    %   LsTLs: L_s' * L_s (gpuArray, Nsources x Nsources)
    %   LtLtT: L_t * L_t' (gpuArray, T x T)
    %   Nsources: Number of sources
    %   T: Number of time points
    %
    % Output:
    %   y: Result of M * x (gpuArray)

    % Reshape x into a matrix (Nsources x T)
    X = reshape(x, [Nsources, T]);

    % Compute M * x = (ATA + LsTLs) * X + X * LtLtT
    Y = (ATA + LsTLs) * X + X * LtLtT;

    % Reshape back into a vector
    y = Y(:);
end
function y = soft_threshold_gpu(x, tau)
    % soft_threshold_gpu: Applies soft-thresholding to a GPU array.
    %
    % Inputs:
    %   x: Input array (gpuArray)
    %   tau: Threshold value (scalar)
    %
    % Output:
    %   y: Thresholded array (gpuArray)

    y = sign(x) .* max(abs(x) - tau, 0);
end