function X = stabilized_solver_L1_L1_L1(B, A, L_s, lambda_s, lambda, rho, max_iter, tol)
    %ADMM for ‖B-AX‖₁ + λₛ‖LₛX‖₁ + λ‖X‖₁
    %L(X, Z₁, Z₂, Z₃, U₁, U₂, U₃) = ‖Z₁‖₁ + λₛ‖Z₂‖₁ + λ‖Z₃‖₁ + (ρ/2)(‖B - AX - Z₁ + U₁‖² + ‖LₛX - Z₂ + U₂‖² + ‖X - Z₃ + U₃‖²)
    [N, T] = size(B);
    M = size(A, 2);
    verbose = true;

    % Initialize variables
    X = A' * B * 0.01;  % Warm start
    Z1 = zeros(N, T);   % For ‖B - AX‖₁
    Z2 = zeros(M, T);   % For ‖LₛX‖₁
    Z3 = zeros(M, T);   % For ‖X‖₁
    U1 = zeros(N, T);   % Dual variable for Z1
    U2 = zeros(M, T);   % Dual variable for Z2
    U3 = zeros(M, T);   % Dual variable for Z3

    % Precompute matrices (with small regularization for stability)
    AtA = A' * A + 1e-8 * eye(M);
    LsTLs = L_s' * L_s + 1e-8 * eye(M);
    I = eye(M);

    if verbose
        fprintf('ADMM for ‖B-AX‖₁ + λₛ‖LₛX‖₁ + λ‖X‖₁\n');
        fprintf('============================================================\n');
        fprintf('N = %d (channels), M = %d (sources), T = %d (time points)\n', N, M, T);
        fprintf('λₛ = %.3f, λ = %.3f, ρ = %.3f, max_iter = %d, tol = %.1e\n', lambda_s, lambda, rho, max_iter, tol);
        fprintf('------------------------------------------------------------\n');
        fprintf('Iter\tPrimal Res\tDual Res\tObjective\tRho\n');
        fprintf('------------------------------------------------------------\n');
    end

    for iter = 1:max_iter
        X_prev = X;
        Z1_prev = Z1;
        Z2_prev = Z2;
        Z3_prev = Z3;

        % --- X-update (Least Squares) ---
        % Solve: (A'A + ρ Lₛ'Lₛ + ρI) X = A'(B - Z1 - U1) + ρ Lₛ'(Z2 + U2) + ρ (Z3 + U3)
        rhs = A'*(B - Z1 - U1) + rho * (L_s'*(Z2 + U2) + (Z3 + U3));
        X = (AtA + rho * LsTLs + rho * I) \ rhs;

        % --- Z1-update (Soft-thresholding for ‖B - AX‖₁) ---
        residual1 = B - A*X - U1;
        Z1 = soft_threshold(residual1, 1/rho);

        % --- Z2-update (Soft-thresholding for ‖LₛX‖₁) ---
        residual2 = L_s*X - U2;
        Z2 = soft_threshold(residual2, lambda_s/rho);

        % --- Z3-update (Soft-thresholding for ‖X‖₁) ---
        residual3 = X - U3;
        Z3 = soft_threshold(residual3, lambda/rho);

        % --- Dual Updates ---
        U1 = U1 + (B - A*X - Z1);
        U2 = U2 + (L_s*X - Z2);
        U3 = U3 + (X - Z3);

        % --- Residuals & Convergence Check ---
        primal_res = norm(B - A*X - Z1, 'fro') + norm(L_s*X - Z2, 'fro') + norm(X - Z3, 'fro');
        dual_res = rho * (norm(A'*(Z1 - Z1_prev), 'fro') + norm(L_s'*(Z2 - Z2_prev), 'fro') + norm(Z3 - Z3_prev, 'fro'));

        % Adaptive ρ adjustment (optional)
        if primal_res > 10 * dual_res
            rho = rho * 1.5;
        elseif dual_res > 10 * primal_res
            rho = rho / 1.5;
        end

        % Objective: ‖B - AX‖₁ + λₛ‖LₛX‖₁ + λ‖X‖₁
        obj = sum(abs(B - A*X), 'all') + lambda_s * sum(abs(L_s*X), 'all') + lambda * sum(abs(X), 'all');

        if verbose
            fprintf('%4d\t%.3e\t%.3e\t%.3e\t%.3f\n', iter, primal_res, dual_res, obj, rho);
        end

        % Check convergence
        if primal_res < tol && dual_res < tol
            if verbose
                fprintf('------------------------------------------------------------\n');
                fprintf('Converged at iteration %d\n', iter);
            end
            break;
        end
    end
end

function X = soft_threshold(X, threshold)
    X = sign(X) .* max(abs(X) - threshold, 0);
end