function J_reconstructed = solve_inverse_problem(B, A, L_s, L_t, lambda_s, lambda_t, T, tol, max_iter)
    % Solves the inverse problem for EEG source localization using some variation of Tikhonov regularization.
    % The regularized problem is formulated as
    % min_{X} {​∥B−A⋅X∥_{2}^{2}​+λ_s​∥L_s​⋅X∥_{2}^{2}​+λ_t​∥X⋅L_{t}^{T}∥​_{2}^{2}}
    % ​∥B−A⋅X∥_{2}^{2}  is the data fidelity term, ensuring the solution fits the measured data.
    % λ_s​∥L_s​⋅X∥_{2}^{2} is the spatial regularization term, encouraging smoothness in space.
    % λ_t​∥X⋅L_{t}^{T}∥​_{2}^{2} is the temporal regularization term, encouraging smoothness in time.
    % L_s​ is the spatial Laplacian matrix (size Nsources×Nsources​).
    % L_t​ is the temporal Laplacian matrix (size T×T).
    % λ_s​ and λt​ ar_e regularization parameters controlling the strength of spatial and temporal smoothness, respectively.
    % 
    % conjugate gradient method:
    % The source activity matrix X is reshaped into a vector x of size Nsources⋅T×1
    % The EEG data B is reshaped into a vector Bvec​ of size Nch⋅T×1
    % The system matrix M is defined as:
    % M=A^TA+λ_s​L_s​+λ_t​L_t​
    % This matrix is applied to xx using the function apply_system_matrix
    % 
    % The CG method is used to solve:
    % M⋅x=A^TB_vec​​
    %
    % Inputs:
    %   B: Measured EEG data (Nch x T)
    %   A: Leadfield matrix (Nch x Nsources)
    %   L_s: Spatial Laplacian matrix (Nsources x Nsources)
    %   L_t: Temporal Laplacian matrix (T x T)
    %   lambda_s: Spatial regularization parameter
    %   lambda_t: Temporal regularization parameter
    %   T: Number of time points
    %   tol: Tolerance for the Conjugate Gradient solver
    %   max_iter: Maximum number of iterations for the Conjugate Gradient solver
    %
    % Output:
    %   J_reconstructed: Reconstructed source activity (Nsources x T)

    % Ensure B has the correct size (Nch x T)
    if size(B, 2) ~= T
        error('B must have T columns (time points).');
    end

    % Number of sources
    Nsources = size(A, 2);

    % Reshape EEG data into a vector
    B_vec = B(:); % Vectorized EEG data (Nch * T x 1)

    % Compute the right-hand side for the linear system
    A_transpose_B = A' * B; % (Nsources x T)
    A_transpose_B_vec = A_transpose_B(:); % Vectorized (Nsources * T x 1)

    % Use Conjugate Gradient (CG) solver to solve (A' * A + lambda_s * L_s + lambda_t * L_t) * J_vec = A' * B_vec
    J_vec = pcg(@(x) apply_system_matrix(x, A, L_s, L_t, lambda_s, lambda_t, T), A_transpose_B_vec, tol, max_iter);

    % Reshape back to (Nsources x T)
    J_reconstructed = reshape(J_vec, Nsources, T);
end

% Helping function for system matrix-vector multiplication
function y = apply_system_matrix(x, A, L_s, L_t, lambda_s, lambda_t, T)
    % Reshape vector into (Nsources x T)
    X = reshape(x, [], T);

    % Compute A' * (A * X)
    AX = A' * (A * X);

    % Compute spatial regularization term (lambda_s * L_s * X)
    LsX = lambda_s * (L_s * X);

    % Compute temporal regularization term (lambda_t * X * L_t')
    LtX = lambda_t * (X * L_t');

    % Combine terms
    y = AX + LsX + LtX;

    % Reshape back into a vector
    y = y(:);
end