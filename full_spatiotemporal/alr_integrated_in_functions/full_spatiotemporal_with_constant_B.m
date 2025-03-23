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
active_sources = 8; % Number of active sources
Xsimulated = zeros(Nsources, T); % Initialize source activity matrix

% Predefine which sources are active and keep the same sources active over time
active_source_indices = 1:active_sources; % You can select specific sources as well
activation_values = rand(active_sources, 1); % Fixed activation values

% Assign constant activation to these sources over all time points
for t = 1:T
    Xsimulated(active_source_indices, t) = activation_values; % Keep the same activation over time
end

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

% Define function handle for system matrix-vector multiplication
function y = apply_system_matrix(x, A, L_s, L_t, lambda_s, lambda_t, T)
    X = reshape(x, [], T); % Reshape vector into (Nsources x T)
    AX = A' * (A * X); % Efficient multiplication with A
    LsX = lambda_s * (L_s * X); % Spatial smoothing
    LtX = lambda_t * (X * L_t'); % Temporal smoothing (transpose L_t for correct dimensions)
    y = AX + LsX + LtX; % System output
    y = y(:); % Return as vector
end

% Solve the full spatio-temporal inverse problem
lambda_s = 0.1e-8; % Spatial regularization parameter
lambda_t = 0.1e-8; % Temporal regularization parameter

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
tol = 1e-5; % Convergence tolerance
max_iter = 1500; % Maximum iterations
J_vec = pcg(@(x) apply_system_matrix(x, A, L_s, L_t, lambda_s, lambda_t, T), A_transpose_B_vec, tol, max_iter);

% Reshape back to (Nsources x T)
J_reconstructed = reshape(J_vec, Nsources, T);

% Display results
% Plot for the inverse problem
figure;
hold on;

% Plot all dipoles in 3D (the inverse problem ones)
all_dipoles = DipoleField.pos(good_dipoles, :); % Positions of all dipoles (N x 3)
dipole_plot = scatter3(all_dipoles(:, 1), all_dipoles(:, 2), all_dipoles(:, 3), 10, 'k', 'filled'); % Plot all dipoles in black
xlabel('x'); ylabel('y'); zlabel('z'); title('3D Dipole Positions with Active Sources (Inverse Problem)');
axis equal; grid on;

% Define a threshold for active dipoles
threshold = 0.8 * max(J_reconstructed(:)); % Adjust threshold as needed

% Map source indices to dipole indices
% Since each dipole has 3 orientations, divide by 3 and round up
dipole_indices = ceil((1:Nsources) / 3); % Maps each source index to a dipole index

% Initialize a list to track active dipoles
active_dipoles_prev = []; % Stores previously active dipoles

% Loop through each time point and highlight active dipoles
for t = 1:T
    % Find active sources at time t
    active_sources = find(J_reconstructed(:, t) > threshold);
    
    % Map active sources to active dipoles
    active_dipoles = unique(dipole_indices(active_sources)); % Get unique dipole indices
    
    % Turn previously active dipoles black (if they are no longer active)
    if ~isempty(active_dipoles_prev)
        % Find dipoles that were active previously but are not active now
        inactive_dipoles = setdiff(active_dipoles_prev, active_dipoles);
        
        % Turn these dipoles black
        if ~isempty(inactive_dipoles)
            scatter3(all_dipoles(inactive_dipoles, 1), all_dipoles(inactive_dipoles, 2), all_dipoles(inactive_dipoles, 3), 10, 'k', 'filled');
        end
    end
    
    % Highlight active dipoles in red
    scatter3(all_dipoles(active_dipoles, 1), all_dipoles(active_dipoles, 2), all_dipoles(active_dipoles, 3), 10, 'r', 'filled');
    
    % Update the list of previously active dipoles
    active_dipoles_prev = active_dipoles;
    
    % Add title with time point information
    title(sprintf('Active Dipoles (inverse) at Time Point %d/%d', t, T));
    
    % Pause to visualize the change
    pause(0.1); % Adjust pause duration as needed
end
hold off;

% Plot for the forward problem
figure;
hold on;

% Plot all dipoles in 3D (the direct problem ones)
all_dipoles = DipoleField.pos(good_dipoles, :); % Positions of all dipoles (N x 3)
dipole_plot = scatter3(all_dipoles(:, 1), all_dipoles(:, 2), all_dipoles(:, 3), 10, 'k', 'filled'); % Plot all dipoles in black
xlabel('x'); ylabel('y'); zlabel('z'); title('3D Dipole Positions with Active Sources (Forward Problem)');
axis equal; grid on;

% Define a threshold for active dipoles
threshold = 0.01 * max(Xsimulated(:)); % Adjust threshold as needed

% Map source indices to dipole indices
% Since each dipole has 3 orientations, divide by 3 and round up
dipole_indices = ceil((1:Nsources) / 3); % Maps each source index to a dipole index

% Initialize a list to track active dipoles
active_dipoles_prev = []; % Stores previously active dipoles

% Loop through each time point and highlight active dipoles
for t = 1:T
    % Find active sources at time t
    active_sources = find(Xsimulated(:, t) > threshold);
    
    % Map active sources to active dipoles
    active_dipoles = unique(dipole_indices(active_sources)); % Get unique dipole indices
    
    % Turn previously active dipoles black (if they are no longer active)
    if ~isempty(active_dipoles_prev)
        % Find dipoles that were active previously but are not active now
        inactive_dipoles = setdiff(active_dipoles_prev, active_dipoles);
        
        % Turn these dipoles black
        if ~isempty(inactive_dipoles)
            scatter3(all_dipoles(inactive_dipoles, 1), all_dipoles(inactive_dipoles, 2), all_dipoles(inactive_dipoles, 3), 10, 'k', 'filled');
        end
    end
    
    % Highlight active dipoles in red
    scatter3(all_dipoles(active_dipoles, 1), all_dipoles(active_dipoles, 2), all_dipoles(active_dipoles, 3), 10, 'r', 'filled');
    
    % Update the list of previously active dipoles
    active_dipoles_prev = active_dipoles;
    
    % Add title with time point information
    title(sprintf('Active Dipoles (direct) at Time Point %d/%d', t, T));
    
    % Pause to visualize the change
    pause(0.1); % Adjust pause duration as needed
end
hold off;