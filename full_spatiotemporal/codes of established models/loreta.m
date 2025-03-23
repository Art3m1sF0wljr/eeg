function X = loreta(A, b, lambda)
    % LORETA Implementation for EEG Source Localization
    % INPUTS:
    %   A      - Lead field matrix (m x n)
    %   b      - Measured EEG data (m x 1)
    %   lambda - Regularization parameter (scalar)
    % OUTPUT:
    %   X      - Estimated source activity (n x 1)

    % Compute the Laplacian matrix L (approximate if not given)
    n = size(A, 2); % Number of sources
    L = laplacian_matrix(n); % Generate an n x n Laplacian matrix

    % Compute the inverse solution using LORETA formula
    X = (A' * A + lambda * (L' * L)) \ (A' * b);
end

function L = laplacian_matrix(n)
    % Generate an approximate n x n discrete Laplacian matrix for smoothness constraint
    % Simple second-order differences with zero padding

    L = -2 * eye(n) + diag(ones(n-1,1), 1) + diag(ones(n-1,1), -1);
    L(1,1) = -1; L(n,n) = -1; % Boundary conditions
end
% Example Usage:
m = 64; % Number of EEG electrodes
n = 100; % Number of brain source locations

A = randn(m, n); % Simulated lead field matrix
b = randn(m, 1); % Simulated EEG measurements
lambda = 0.1; % Regularization parameter (tune this for better results)

X = loreta(A, b, lambda);

% Plot the estimated source distribution
figure;
plot(X);
title('Estimated Source Activity using LORETA');
xlabel('Source Index');
ylabel('Activation');
