clear all; close all;

% Load DipoleField and electrode positions
load('DipoleField');

% Load EEG data from EDF file
[fname, fdir] = uigetfile('*.edf', 'Select the EEG file');
[hdr, recrd] = edfread([fdir fname]);
ch_name = hdr.label;

% Match EEG channels with electrode positions
k = 1;
for i = 1:length(ch_name)
    for j = 1:97 % Assuming 97 electrodes
        if (strcmp(ch_name{i}, elec.label{j}))
            good_ch(k) = i; % Store the index of the matching channel
            good(k) = j; % Store the index of the matching electrode
            position(k, :) = elec.elecpos(j, :); % Store the electrode's position
            k = k + 1;
        end
    end
end
clear ch_name; ch_name = elec.label(good);
record = recrd(good_ch, :); % Extract matching EEG channels
Nch = size(record, 1); % Number of channels
fs = hdr.samples(10); % Sampling frequency
N = fs;

% Use only the first 1000 time samples
T = 1000; % Number of time points to process
record = record(:, 1:T); % Truncate the EEG data

% Associate each dipole with the nearest electrode
pos_good_dipoles = DipoleField.pos(DipoleField.inside, :);
DistMat = pdist2(pos_good_dipoles, position);
[m, I] = min(DistMat');

% Create interpolation surface for electrode positions
S = scatteredInterpolant(elec.elecpos(:, 1), elec.elecpos(:, 2), elec.elecpos(:, 3), 'natural');
minX = min(elec.elecpos(:, 1)); maxX = max(elec.elecpos(:, 1));
minY = min(elec.elecpos(:, 2)); maxY = max(elec.elecpos(:, 2));
[xx, yy] = meshgrid(linspace(minX, maxX, 20), linspace(minY, maxY, 20));
zz = S(xx, yy);

% Check which points are inside the convex hull of the head
tess = convhulln(elec.elecpos(:, 1:2));
in = inhull([xx(:) yy(:)], elec.elecpos(:, 1:2), tess);
bad = find(in == 0);
zz(bad) = nan;

% Construct Leadfield Matrix A
Nsources = 3 * sum(DipoleField.inside); % Total number of dipole components
good_dipoles = find(DipoleField.inside == 1); % Indices of active dipoles
A = zeros(Nch, Nsources);
n = 1;
for i = 1:Nsources / 3
    A(:, n:n+2) = DipoleField.leadfield{good_dipoles(i)}(good, :); % Store leadfield for each dipole
    n = n + 3;
end

% Solve the inverse problem using the provided EEG data
lambda_s = 0.1e-8; % Spatial regularization parameter
lambda_t = 0.1e-8; % Temporal regularization parameter
tol = 3e-5; % Convergence tolerance
max_iter = 1500; % Maximum iterations

dipole_positions = DipoleField.pos(good_dipoles, :); % N_dipoles x 3 matrix
sigma = 10; % Scaling parameter
neighborhood_threshold = 10; 

% Spatial Laplacian (Nsources x Nsources)
%L_s = construct_spatial_laplacian(dipole_positions, sigma, neighborhood_threshold);
L_s = speye(Nsources); % Use identity matrix for simplicity (replace with actual spatial Laplacian if available)


% Temporal smoothness matrix (T x T)
L_t = diag(ones(T-1, 1), 1) + diag(ones(T-1, 1), -1) - 2 * eye(T); % Temporal smoothness matrix, approximates second derivative thru finite differences

% Solve the inverse problem
J_reconstructed = solve_inverse_problem(record, A, L_s, L_t, lambda_s, lambda_t, T, tol, max_iter);




% Display results

% Reconstruct the estimated EEG signal
est = A * J_reconstructed;

% Plot the raw EEG and estimated EEG signals
figure;
hl(1) = subplot(211);
mostra_segnali(record / 6 / std(record(:)), 'k', '', fs);
mostra_segnali(est / 6 / std(record(:)), 'r', '', fs);
title('EEG Signal');
legend('Raw EEG', 'Estimated EEG');
set(gca, 'YTick', [1:Nch]); set(gca, 'YTickLabel', ch_name);

% Selective channel reconstruction
for ch = 1:Nch
    select_dipoles = find(I == ch);
    select_col = sort([3 * (select_dipoles - 1) + 1, 3 * (select_dipoles - 1) + 2, 3 * (select_dipoles - 1) + 3]);
    vv = (A(:, select_col) * J_reconstructed(select_col, :)); % Reconstruct signal for selective dipoles
    sig(ch, :) = vv(ch, :); % Store the reconstructed signal for the channel
end

% Plot the selective channel reconstruction
hl(2) = subplot(212);
mostra_segnali(sig / 6 / std(sig(:)), 'k', '', fs);
set(gca, 'YTick', [1:Nch]); set(gca, 'YTickLabel', ch_name);
title('Selective Channels: Signals from Closest Dipoles (No Crosstalk)');
linkaxes(hl, 'xy');


% Plot for the inverse problem
figure;
hold on;

% Plot all dipoles in 3D (the inverse problem ones)
all_dipoles = DipoleField.pos(good_dipoles, :); % Positions of all dipoles (N x 3)
dipole_plot = scatter3(all_dipoles(:, 1), all_dipoles(:, 2), all_dipoles(:, 3), 10, 'k', 'filled'); % Plot all dipoles in black
xlabel('x'); ylabel('y'); zlabel('z'); title('3D Dipole Positions with Active Sources (Inverse Problem)');
axis equal; grid on;

% Define a threshold for active dipoles
threshold = 0.4 * max(J_reconstructed(:)); % Adjust threshold as needed

% Map source indices to dipole indices
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
    pause(0.01 * 100 / T); % Adjust pause duration as needed
end
hold off;