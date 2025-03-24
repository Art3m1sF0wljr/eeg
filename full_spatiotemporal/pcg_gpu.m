function [x, flag, relres, iter] = pcg_gpu(A, b, tol, max_iter)
    % pcg_gpu: GPU-accelerated Preconditioned Conjugate Gradient solver.
    %
    % Inputs:
    %   A: Function handle for matrix-vector multiplication (A(x) = M * x)
    %   b: Right-hand side vector (gpuArray)
    %   tol: Tolerance for convergence
    %   max_iter: Maximum number of iterations
    %
    % Outputs:
    %   x: Solution vector (gpuArray)
    %   flag: Convergence flag (0: converged, 1: max iterations reached)
    %   relres: Relative residual norm
    %   iter: Number of iterations performed

    % Initialize variables
    x = gpuArray.zeros(size(b)); % Initial guess
    r = b - A(x); % Residual
    p = r; % Search direction
    rsold = gather(r' * r); % Scalar: r' * r (computed on GPU, gathered to CPU)

    % PCG iterations
    for iter = 1:max_iter
        Ap = A(p); % Matrix-vector multiplication
        alpha = rsold / gather(p' * Ap); % Step size (scalar, computed on GPU, gathered to CPU)
        x = x + alpha * p; % Update solution
        r = r - alpha * Ap; % Update residual
        rsnew = gather(r' * r); % New residual norm (scalar, computed on GPU, gathered to CPU)

        % Check for convergence
        if sqrt(rsnew) < tol
            flag = 0; % Converged
            relres = sqrt(rsnew);
            return;
        end

        % Update search direction
        p = r + (rsnew / rsold) * p;
        rsold = rsnew;
    end

    % If max iterations reached
    flag = 1; % Did not converge
    relres = sqrt(rsold);
end