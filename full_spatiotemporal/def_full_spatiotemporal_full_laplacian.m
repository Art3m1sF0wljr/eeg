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

%close points have similar number.
%this is such that in Xsimulated the nodes that light up are 6 at step 0, and at each step only points that are close, say in the neighbourhood of 10 points, to the ones already light up do light up with a probability proportional to the distance. also the dipoles already light up at the previous step have probability/6 that they turn off
% Simulate activation of sources over T time points
% Parameters for EEG data generation
T = 100; % Number of time points
active_sources = 6; % Number of active sources

%for smootly_varying_B_numerical_center
neighborhood_size = 10; % Neighborhood size for source activation
base_probability = 0.001; % Base probability of a source lighting up
space_smooth_factor = 0.2; % Spatial smoothness factor
%for constant_B
active_source_indices = 1:active_sources; % First 8 sources are active
activation_values = rand(active_sources, 1); % Random activation values for the active sources

% Generate EEG data using the function 1 or 2
% for constant_B optimal threshold for inverse is 0.8
%[B, Xsimulated] = smootly_varying_B_numerical_center(A, good_dipoles, T, active_sources, neighborhood_size, base_probability, space_smooth_factor);
%[B, Xsimulated] = constant_B(A, active_source_indices, activation_values, T);
[B, Xsimulated] = professor_generated_B(A, good_dipoles, T, active_sources, space_smooth_factor);

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



% Solve the inverse problem using the provided function
lambda_s = 0.1; % Spatial regularization parameter 0.1e-8 for L2
lambda_t = 0.1; % Temporal regularization parameter 0.1e-8 for L2
tol = 1e-4; % Convergence tolerance  3e-5 for L2
max_iter = 1500; % Maximum iterations
rho = 1.0;      % ADMM penalty parameter

% Spatial Laplacian (Nsources x Nsources)
% Assuming dipole_positions is a Nsources x 3 matrix containing the positions of the dipoles
dipole_positions = DipoleField.pos(good_dipoles, :); % N_dipoles x 3 matrix
sigma = 10; % Scaling parameter
neighborhood_threshold = 10; 
verbose = 2;  %0,1,2
L_s = construct_spatial_laplacian(dipole_positions, sigma, neighborhood_threshold);

% Temporal smoothness matrix (T x T)
L_t = diag(ones(T-1,1), 1) + diag(ones(T-1,1), -1) - 2*eye(T); % Temporal smoothness matrix, approximates second derivative thru finite differences

% Solve the inverse problem
%J_reconstructed = solve_inverse_problem(B, A, L_s, L_t, lambda_s, lambda_t, T, tol, max_iter);
%J_reconstructed = ADMM_L1_minimization(B, A, L_s, L_t, lambda_s, lambda_t, rho, max_iter, tol, verbose);
J_reconstructed = ADMM_L1_minimization_GPU(B, A, L_s, L_t, lambda_s, lambda_t, rho, max_iter, tol, verbose);

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
threshold = 0.5 * max(J_reconstructed(:)); % Adjust threshold as needed

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
    pause(0.1*100/T); % Adjust pause duration as needed
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
    pause(0.1*100/T); % Adjust pause duration as needed
end
hold off;