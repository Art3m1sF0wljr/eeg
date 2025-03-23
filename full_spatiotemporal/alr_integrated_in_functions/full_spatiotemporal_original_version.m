clear all; close all;
load('DipoleField'); 
Nsources = 3 * sum(DipoleField.inside);

% Create interpolation surface for electrode positions
S = scatteredInterpolant(elec.elecpos(:,1), elec.elecpos(:,2), elec.elecpos(:,3), 'natural');
minX = min(elec.elecpos(:,1)); maxX = max(elec.elecpos(:,1));
minY = min(elec.elecpos(:,2)); maxY = max(elec.elecpos(:,2));
[xx, yy] = meshgrid(linspace(minX, maxX, 20), linspace(minY, maxY, 20));
zz = S(xx, yy);

% Check which points are inside the convex hull of the head
tess = convhulln(elec.elecpos(:,1:2));
in = inhull([xx(:) yy(:)], elec.elecpos(:,1:2), tess);
bad = find(in == 0);
zz(bad) = nan;

% Simulate EEG leadfield matrix
good_dipoles = find(DipoleField.inside == 1);
Nch = length(DipoleField.label);
A = zeros(Nch, Nsources);
n = 1;
for i = 1:Nsources/3
    A(:, n:n+2) = DipoleField.leadfield{good_dipoles(i)};
    n = n + 3;
end

% Simulate activation of sources over T time points
T = 100; % Number of time points
active_sources = 10; % Number of active sources
Xsimulated = zeros(Nsources, T); % Initialize source activity matrix

% Temporal smoothness function (e.g., Gaussian envelope)
time_smooth = exp(-linspace(-2, 2, T).^2); % Gaussian envelope for smooth variation

% Simulate source activity over time with smooth variations
for t = 1:T
    temp_activity = rand(active_sources, 1); % Random activation for active sources
    Xsimulated(1:active_sources, t) = temp_activity .* time_smooth(t); % Apply temporal smoothing
end

% Spatial smoothness: smooth source activity over space
space_smooth_factor = 0.2; % Spatial smoothness factor (higher means more smoothing)
Xsimulated = smooth_source_activity(Xsimulated, space_smooth_factor); % Apply spatial smoothing

% Simulate EEG data over T time points
B = A * Xsimulated; % Forward EEG computation (Nch x T)

% Display EEG scalp potentials at the first time point
F = scatteredInterpolant(elec.elecpos(:,1), elec.elecpos(:,2), elec.elecpos(:,3), B(:, 1), 'natural');
surf(xx, yy, zz, F(xx, yy, zz), 'EdgeColor', 'none'); alpha(.6);
xlabel('x'); ylabel('y'); zlabel('z'); axis equal; hold on;

% Plot activated sources at the first time point
qq = find(Xsimulated(:, 2) > 0); % Active sources at t = 2
for Num = 1:length(qq)
    meanA = Xsimulated(qq(Num), 1);
    Num_dipole = round(qq(Num) / 3 + 1);
    color_level = round(100 * (.9 - .9 * meanA / max(Xsimulated(:, 1)))) / 100;
    plot3(DipoleField.pos(good_dipoles(Num_dipole),1), DipoleField.pos(good_dipoles(Num_dipole),2), DipoleField.pos(good_dipoles(Num_dipole),3), 'o', 'color', color_level * [1 1 1], 'MarkerSize', 1 + 5 * meanA / max(Xsimulated(:, 1)), 'LineWidth', 1 + 2 * meanA / max(Xsimulated(:, 1)));
end

% Solve the full spatio-temporal inverse problem
lambda_s = 0.001; % Spatial regularization parameter
lambda_t = 0.001; % Temporal regularization parameter

% Ensure B is of size (Nch x T)
if size(B, 2) ~= T
    error('B must have T columns (time points).');
end

% Spatial Laplacian (Nsources x Nsources)
L_s = speye(Nsources); % Use identity matrix for simplicity (replace with actual spatial Laplacian if available)

% Temporal smoothness matrix (T x T)
L_t = diag(ones(T-1,1), 1) + diag(ones(T-1,1), -1) - 2*eye(T); % Temporal smoothness matrix

% Reshape EEG data into a vector
B_vec = B(:); % Vectorized EEG data (Nch * T x 1)

% Compute the right-hand side for the linear system
A_transpose_B = A' * B; % (Nsources x T)
A_transpose_B_vec = A_transpose_B(:); % Vectorized (Nsources * T x 1)

% Use Conjugate Gradient (CG) solver to solve (A' * A + L_s + L_t) * J_vec = A' * B_vec
tol = 1e-3; % Convergence tolerance
max_iter = 500; % Maximum iterations
J_vec = pcg(@(x) apply_system_matrix(x, A, L_s, L_t, lambda_s, lambda_t, T), A_transpose_B_vec, tol, max_iter);

% Reshape back to (Nsources x T)
J_reconstructed = reshape(J_vec, Nsources, T);

% Display results
figure;
subplot(2,1,1); imagesc(Xsimulated); title('True Source Activity'); colorbar;
subplot(2,1,2); imagesc(J_reconstructed); title('Reconstructed Source Activity'); colorbar;

% Define function handle for system matrix-vector multiplication
function y = apply_system_matrix(x, A, L_s, L_t, lambda_s, lambda_t, T)
    X = reshape(x, [], T); % Reshape vector into (Nsources x T)
    AX = A' * (A * X); % Efficient multiplication with A
    LsX = lambda_s * (L_s * X); % Spatial smoothing
    LtX = lambda_t * (X * L_t'); % Temporal smoothing (transpose L_t for correct dimensions)
    y = AX + LsX + LtX; % System output
    y = y(:); % Return as vector
end

% Function for smoothing source activity over space
function smoothed_activity = smooth_source_activity(activity, factor)
    smoothed_activity = activity;
    for t = 1:size(activity, 2)
        smoothed_activity(:, t) = smoothdata(activity(:, t), 'movmean', factor);
    end
end
