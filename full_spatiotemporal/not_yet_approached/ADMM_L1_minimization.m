function X_reconstructed = ADMM_L1_minimization(B, A, L_s, L_t, lambda_s, lambda_t, rho, max_iter, tol, verbose)
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
    %   verbose: Verbosity level (0: no output, 1: print residuals, 2: print all variables)
    %
    % Output:
    %   X_reconstructed: Reconstructed source activity (Nsources x T)
    % Dimensions
    [Nch, T] = size(B);
    Nsources = size(A, 2);
	
	% Force sparse storage (even if matrices appear full)
    A = sparse(double(A));
    L_s = sparse(double(L_s));
    L_t = sparse(double(L_t));
    B = double(B);  % Keep B dense if it's not extremely large

    % Initialize variables
    X = zeros(Nsources, T);  % Source activity
    %Z1 = zeros(Nch, T);      % Auxiliary variable for data fidelity
    %Z2 = zeros(Nsources, T); % Auxiliary variable for spatial regularization
    %Z3 = zeros(Nsources, T); % Auxiliary variable for temporal regularization
    %U1 = zeros(Nch, T);      % Dual variable for data fidelity
    %U2 = zeros(Nsources, T); % Dual variable for spatial regularization
    %U3 = zeros(Nsources, T); % Dual variable for temporal regularization
    [Z1, Z2, Z3] = deal(zeros(size(B)), zeros(Nsources, T), zeros(Nsources, T));
    [U1, U2, U3] = deal(zeros(size(B)), zeros(Nsources, T), zeros(Nsources, T));
	
    % Store previous Z values for dual residual calculation
    Z1_prev = Z1;
    Z2_prev = Z2;
    Z3_prev = Z3;

    % Precompute matrices for X update
	ATA = sparse(A' * A);
    LsTLs = sparse(L_s' * L_s);
    LtLtT = sparse(L_t * L_t');
	
    %ATA = A' * A;  % Size: Nsources x Nsources
    %LsTLs = L_s' * L_s;  % Size: Nsources x Nsources
    %LtLtT = L_t * L_t';  % Size: T x T

    % Preconditioner setup (diagonal preconditioner)
    %diag_ATA_LsTLs = diag(ATA + LsTLs); % Nsources x 1 vector
    %diag_LtLtT = diag(LtLtT); % T x 1 vector
    % Create efficient diagonal preconditioner function
    %preconditioner = @(x) x ./ (kron(ones(T,1), diag_ATA_LsTLs) + kron(diag_LtLtT, ones(Nsources,1)));
	% =============================================
    % NEW: Optimized Preconditioner Setup
    % =============================================
    
    % Spatial preconditioner (incomplete Cholesky)
    % Ensure positive definiteness
    M_spatial = ATA + LsTLs;
    min_eig = eigs(M_spatial, 1, 'smallestreal');
    if min_eig <= 0
        reg_param = abs(min_eig) + 1e-3;
        M_spatial = M_spatial + reg_param*speye(Nsources);
    end
    
    % 2. Use modified ichol with increased diagonal compensation
    opts.diagcomp = 1e-3;  % Additional diagonal boost
    opts.type = 'ict';
    opts.droptol = 1e-2;
    %M_spatial_ichol = ichol(M_spatial, opts);
	
	try
		M_spatial_ichol = ichol(M_spatial, opts);
	catch
		warning('ichol failed, using diagonal preconditioner');
		D = diag(M_spatial);
		D(D < 1e-6) = 1e-6;  % Ensure positive diagonal
		M_spatial_ichol = struct('L', spdiags(sqrt(D), 0, Nsources, Nsources), ...
								'U', spdiags(sqrt(D), 0, Nsources, Nsources));
	end	
	
    % Temporal preconditioner (regularized LtLtT is pentadiagonal)
    M_temporal = sparse(LtLtT + 1e-4*speye(T));
    
    % Create efficient block-diagonal preconditioner function
    preconditioner = @(x) block_diagonal_solve(M_spatial_ichol, M_temporal, x, Nsources, T);
    

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

        % Solve M * X_vec = RHS using PCG with preconditioner
        max_iter_pcg = min(5000, max_iter*10);  % Adjusted max PCG iterations
        [X_vec, flag, relres, pcg_iter] = pcg(...
            @(x) apply_system_matrix_implicit(x, ATA, LsTLs, LtLtT, Nsources, T), ...
            full(RHS(:)), ...% Ensure RHS is dense for pcg
            tol, ...
            max_iter_pcg, ...
            @preconditioner);
		
		if flag ~= 0 && verbose >= 1
            fprintf('PCG warning: flag=%d, relres=%e at iter=%d\n', flag, relres, iter);
        end
		
        % [Rest of the ADMM iterations remain the same]
        
        % Reshape X back to matrix form
        X = reshape(X_vec, [Nsources, T]);
		Z1 = soft_threshold(B - A*X + U1, 1/rho);
        Z2 = soft_threshold(L_s*X + U2, lambda_s/rho);
        Z3 = soft_threshold(X*L_t' + U3, lambda_t/rho);

        % Update Z1 (soft-thresholding for data fidelity)
        %residual1 = B - A * X + U1;
        %Z1 = soft_threshold(residual1, 1 / rho);
        % Update Z2 (soft-thresholding for spatial regularization)
        %residual2 = L_s * X + U2;
        %Z2 = soft_threshold(residual2, lambda_s / rho);
        % Update Z3 (soft-thresholding for temporal regularization)
        %residual3 = X * L_t' + U3;
        %Z3 = soft_threshold(residual3, lambda_t / rho);

        % Update dual variables
        U1 = U1 + (B - A * X - Z1);
        U2 = U2 + (L_s * X - Z2);
        U3 = U3 + (X * L_t' - Z3);

        % Compute primal and dual residuals
        primal_residual = norm(B - A*X - Z1,'fro') + norm(L_s*X - Z2,'fro') + norm(X*L_t' - Z3,'fro');
        dual_residual = rho*(norm(A'*(Z1 - Z1_prev),'fro') + norm(L_s'*(Z2 - Z2_prev),'fro') + norm((Z3 - Z3_prev)*L_t,'fro'));

        % Verbosity: Print progress
        if verbose >= 1
            fprintf('Iteration %d:\n', iter);
            fprintf('  Primal Residual: %e\n', primal_residual);
            fprintf('  Dual Residual: %e\n', dual_residual);
            if verbose >= 2
                fprintf('  X: %s\n', mat2str(X(1:min(3, Nsources), 1:min(3, T)))); % Print a small subset of X
                fprintf('  Z1: %s\n', mat2str(Z1(1:min(3, Nch), 1:min(3, T)))); % Print a small subset of Z1
                fprintf('  Z2: %s\n', mat2str(Z2(1:min(3, Nsources), 1:min(3, T)))); % Print a small subset of Z2
                fprintf('  Z3: %s\n', mat2str(Z3(1:min(3, Nsources), 1:min(3, T)))); % Print a small subset of Z3
                fprintf('  U1: %s\n', mat2str(U1(1:min(3, Nch), 1:min(3, T)))); % Print a small subset of U1
                fprintf('  U2: %s\n', mat2str(U2(1:min(3, Nsources), 1:min(3, T)))); % Print a small subset of U2
                fprintf('  U3: %s\n', mat2str(U3(1:min(3, Nsources), 1:min(3, T)))); % Print a small subset of U3
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

    % Output the reconstructed source activity
    X_reconstructed = X;
end
% =============================================
function y = preconditioner(x)
        try
            X_mat = reshape(x, [Nsources, T]);
            % Temporal solve (exploits banded structure)
            temp = M_temporal' \ X_mat';
            % Spatial solve
            y = M_spatial_ichol \ temp';
            y = M_spatial_ichol' \ y;
            y = y(:);
        catch
            % Fallback to Jacobi preconditioner
            diag_precond = 1 ./ (diag(M_spatial) + kron(ones(T,1), diag(M_temporal)));
            y = x .* diag_precond;
        end
    end
function y = block_diagonal_solve(M_spatial_ichol, M_temporal, x, Nsources, T)
    % Solves (M_spatial ⊗ I + I ⊗ M_temporal) y = x
    X = reshape(x, [Nsources, T]);
    
    % Solve temporal part first (pentadiagonal system)
    Y = M_temporal' \ X';  % Solves M_temporal' * Y = X' efficiently
    
    % Solve spatial part (using ichol factorization)
    % Solve spatial part (forward/backward substitution)
    Y = M_spatial_ichol \ Y';
    Y = M_spatial_ichol' \ Y;
    
    y = Y(:);
end
% Function to apply the system matrix M to a vector x (implicitly)
function y = apply_system_matrix_implicit(x, ATA, LsTLs, LtLtT, Nsources, T)
    % Reshape x into a matrix
    X = reshape(x, [Nsources, T]);

    % Compute M * x without explicitly forming the Kronecker product
    Y = (ATA + LsTLs) * X + X * LtLtT;

    % Reshape back into a vector
    y = Y(:);
end

% Soft-thresholding function
function y = soft_threshold(x, tau)
    y = sign(x) .* max(abs(x) - tau, 0);
end