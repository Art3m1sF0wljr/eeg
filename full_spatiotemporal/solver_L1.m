function X = solver_L1(B, A, lambda, rho, max_iter, tol);
	%min​{∥B−AX∥1​+λ​∥X​∥1​}
    %L(X, Z₁, Z₂, U₁, U₂) = ‖Z₁‖₁ + λ‖Z₂‖₁ + (ρ/2)(‖B - AX - Z₁ + U₁‖² + ‖X - Z₂ + U₂‖²)
    verbose = true;
    [N, T] = size(B);
    M = size(A, 2);
	

    % Normalize the problem
    A_norm = norm(A, 'fro');
    B_norm = norm(B, 'fro');
    A = A/A_norm;
    B = B/B_norm;
    lambda = lambda/(A_norm*B_norm);  % Scale lambda accordingly


	% Initialize variables
    X = zeros(M, T);  % Warm start (better than zeros)
    Z1 = zeros(N, T);  % Auxiliary variable for data term
    Z2 = zeros(M, T);  % Auxiliary variable for position term
    U1 = zeros(N, T);  % Dual variable for data term
    U2 = zeros(M, T);  % Dual variable for position term
	
    % Precompute matrices for X-update
	AtA = A' * A;% + 1e-8 * eye(M);  % Regularized
	I = eye(M);

    % Use Cholesky decomposition for faster solves
    L = chol(AtA + rho*I, 'lower');
	
	% Variables to track best solution
    best_obj = inf;
    best_X = X;
    best_iter = 0;
    best_primal_res = inf;
    best_dual_res = inf;
    best_rho = rho;

	if verbose
        fprintf('ADMM for ‖B-AX‖₁ + λ‖X‖₁\n');
        fprintf('============================================\n');
        fprintf('N = %d (channels), M = %d (sources), T = %d (time)\n', N, M, T);
        fprintf('λ = %.3f, ρ = %.3f, max_iter = %d, tol = %.1e\n', lambda, rho, max_iter, tol);
        fprintf('--------------------------------------------\n');
        fprintf('Iter\tPrimal Res\tDual Res\tObjective\n');
        fprintf('--------------------------------------------\n');
    end
	
	for iter = 1:max_iter
		X_prev = X;
        Z1_prev = Z1;
        Z2_prev = Z2;
		
		% --- Update X (least-squares) ---
        % Solve: (A'A + ρ I)X = A'(B - Z1 - U1) + ρ I(Z2 + U2)
		rhs = A' * (B - Z1 - U1) + rho * (Z2 + U2);
        %X = (AtA + rho * I) \ rhs; % Solve using Cholesky or CG for large M
		X = L' \ (L \ rhs);  % Faster solve using Cholesky

		% --- Update Z1 (data term, L1 proximal) ---
        residual1 = B - A * X + U1;
        Z1 = soft_threshold(residual1, 1/rho);
		
		% --- Update Z2 (position term, L1 proximal) ---
        residual2  = X + U2;
        Z2 = soft_threshold(residual2, lambda/rho);
		
		% --- Dual Updates ---
        U1 = U1 + (B - A*X - Z1);
        U2 = U2 + (X - Z2);
		
		% --- Compute residuals and objective ---
        primal_res = norm(B - A*X - Z1, 'fro') + norm(X - Z2, 'fro');
        dual_res = rho * (norm(A'*(Z1 - Z1_prev), 'fro') + norm(Z2 - Z2_prev, 'fro'));
		
		if primal_res > 10 * dual_res
			rho = rho * 3;   % Increase penalty if primal res is too high
            L = chol(AtA + rho*I, 'lower');
		elseif dual_res > 10 * primal_res
			rho = rho / 3;   % Decrease penalty if dual res is too high
            L = chol(AtA + rho*I, 'lower');
		end
        
        % Objective: ||B - AX||_1 + lambda ||X||_1
        obj = sum(abs(B - A*X), 'all') + lambda * sum(abs(X), 'all');
		
		% Track best solution
        if obj < best_obj
            best_obj = obj;
            best_X = X;
            best_iter = iter;
            best_primal_res = primal_res;
            best_dual_res = dual_res;
            best_rho = rho;
        end
        
        % Adjust rho
        if primal_res > 10 * dual_res
            rho = rho * 3;
            L = chol(AtA + rho*I, 'lower');
        elseif dual_res > 10 * primal_res
            rho = rho / 3;
            L = chol(AtA + rho*I, 'lower');
        end

        if verbose
            fprintf('%4d\t%.3e\t%.3e\t%.3e\t%.3e\n', iter, primal_res, dual_res, obj, rho);
        end

        % Check convergence
        if primal_res < tol && dual_res < tol
            if verbose
                fprintf('--------------------------------------------\n');
                fprintf('Converged at iteration %d\n', iter);
            end
            break;
        end
    end

    % Denormalize the solution
    %X = X * B_norm / A_norm;
	
	best_X = best_X * B_norm / A_norm;
	X=best_X;
    
    % Store best statistics
    %5best_stats = struct(...
    %    'best_iter', best_iter, ...
    %    'best_obj', best_obj * A_norm * B_norm, ...  % Denormalized objective
    %    'best_primal_res', best_primal_res, ...
    %    'best_dual_res', best_dual_res, ...
    %    'best_rho', best_rho);

    if iter == max_iter && verbose
        fprintf('-----------------------------------\n');
        fprintf('Maximum iterations reached\n');
		
		fprintf('Best solution found at iteration %d\n', best_iter);
        fprintf('Best objective: %.3e\n', best_stats.best_obj);
    end
end

function X = soft_threshold(X, threshold)
    X = sign(X) .* max(abs(X) - threshold, 0);
end