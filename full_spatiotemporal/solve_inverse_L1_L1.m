function X = solve_inverse_L1_L1_improved(B, A, L_t, lambda_t, rho, max_iter, tol)
    [N, T] = size(B);
    M = size(A, 2);
    
    % Sparse matrices
    if ~issparse(A), A = sparse(A); end
    if ~issparse(L_t), L_t = sparse(L_t); end
    
    % Precompute matrices and factorizations
    AtA = A' * A;
    I = speye(M);
    [L_U, U_U, P_U] = lu(AtA + rho * I);  % LU factorization for X update
    
    % Warm start with minimum norm solution
    X = A' * (A*A' + 1e-3*speye(N)) \ B;
    Z1 = zeros(N, T);
    Z2 = zeros(M, T);
    U1 = zeros(N, T);
    U2 = zeros(M, T);
    
    % Nesterov acceleration variables
    eta = 1;
    X_hat = X;
    
    for iter = 1:max_iter
        X_prev = X;
        Z1_prev = Z1;
        Z2_prev = Z2;
        
        % --- Update X with Nesterov acceleration ---
        residual_data = B - Z1_prev - U1;
        residual_temp = Z2_prev + U2;
        rhs = A' * residual_data + residual_temp * L_t;
        X = U_U \ (L_U \ (P_U * rhs));
        
        % Nesterov update
        eta_next = (1 + sqrt(1 + 4*eta^2))/2;
        X_hat = X + ((eta - 1)/eta_next) * (X - X_prev);
        eta = eta_next;
        
        % --- Update Z1 (soft thresholding) ---
        residual = B - A * X_hat - U1;
        Z1 = sign(residual) .* max(abs(residual) - 1/rho, 0);
        
        % --- Update Z2 (soft thresholding) ---
        temporal_residual = X_hat * L_t' - U2;
        Z2 = sign(temporal_residual) .* max(abs(temporal_residual) - lambda_t/rho, 0);
        
        % --- Update dual variables ---
        U1 = U1 + (B - A * X_hat - Z1);
        U2 = U2 + (X_hat * L_t' - Z2);
        
        % --- Adaptive rho ---
        primal_res = norm(B - A * X_hat - Z1, 'fro') + norm(X_hat * L_t' - Z2, 'fro');
        dual_res = rho * (norm(A' * (Z1 - Z1_prev), 'fro') + norm((Z2 - Z2_prev) * L_t, 'fro'));
        
        if primal_res > 10 * dual_res
            rho = min(rho * 2, 1e4);
            [L_U, U_U, P_U] = lu(AtA + rho * I);  % Update factorization
        elseif dual_res > 10 * primal_res
            rho = max(rho / 2, 1e-4);
            [L_U, U_U, P_U] = lu(AtA + rho * I);  % Update factorization
        end
        
        % --- Check convergence ---
        if primal_res < tol && dual_res < tol
            break;
        end
    end
end