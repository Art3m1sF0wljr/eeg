close all; clear all; clc;
load('DipoleField');
%load('elec.mat'); % Make sure this contains electrode positions

%% ====================== 1. Load and Preprocess EEG Data ======================
% Load EEG data
[fname, fdir] = uigetfile('*.edf', 'Select the EEG file');
[hdr, record] = edfread([fdir fname]);
% Select only the first second of data
samples_to_keep = hdr.samples(1); % Number of samples in 1 second
% Select the SECOND second (1-2 seconds)
start_sample = 1*samples_to_keep + 1; % Start at beginning of 2nd second
end_sample = min(2*samples_to_keep, size(record,2)); % End at 2 seconds or file end
% Extract the desired segment
record = record(:, start_sample:end_sample);
% Convert to proper FieldTrip structure
data = [];
data.trial = {record};
data.time = {(0:size(record,2)-1)/hdr.samples(1)};
data.label = hdr.label;
data.fsample = hdr.samples(1);

% Simple preprocessing (bandpass filter 1-30 Hz)
cfg = [];
cfg.bpfilter = 'yes';
cfg.bpfreq = [1 45];
data = ft_preprocessing(cfg, data);

%% ====================== 2. Match EEG Channels to Electrode Positions ======================
% Assuming you have standard 10-20 positions if elec.mat isn't available
if ~exist('elec', 'var')
    cfg = [];
    cfg.elec = 'standard_1020.elc'; % Use standard 10-20 system
    elec = ft_read_sens(cfg.elec);
end

good_ch = [];
good_elec = [];
position = [];
k = 1;
for i = 1:length(data.label)
    for j = 1:length(elec.label)
        if strcmpi(data.label{i}, elec.label{j}) % Case insensitive comparison
            good_ch(k) = i;
            good_elec(k) = j;
            position(k,:) = elec.elecpos(j,:);
            k = k + 1;
        end
    end
end

% Select only matched channels
cfg = [];
cfg.channel = data.label(good_ch);
data = ft_selectdata(cfg, data);

%% ====================== 3. Prepare Head Model ======================
cfg = [];
cfg.method = 'singleshell';
cfg.headshape = position;
headmodel = ft_prepare_headmodel(cfg);

%% ====================== 4. Set Up Source Model ======================
sourcemodel = [];
sourcemodel.pos = DipoleField.pos;
sourcemodel.inside = DipoleField.inside;
sourcemodel.leadfield = DipoleField.leadfield;

% Adjust leadfield dimensions to match our data
for i = 1:length(sourcemodel.leadfield)
    if ~isempty(sourcemodel.leadfield{i}) && sourcemodel.inside(i)
        sourcemodel.leadfield{i} = sourcemodel.leadfield{i}(good_elec, :);
    end
end

%% ====================== 5. Prepare Data Matrix ======================
% Get the data matrix (Nchannels × T)
B = data.trial{1};  % This is now your Nchannels × T matrix
%% ====================== 6. Custom Inverse Solution ======================
% Construct leadfield matrix A (Nchannels × 3Nsources)
inside_idx = find(sourcemodel.inside);
n_sources = length(inside_idx);
n_channels = size(B, 1);
n_timepoints = size(B, 2);

A = zeros(n_channels, 3*n_sources);  % 3 components per source
for i = 1:n_sources
    idx = inside_idx(i);
    A(:, (3*i-2):3*i) = sourcemodel.leadfield{idx};
end

% Regularization parameters
lambda = 0.1 * trace(A*A') / n_channels;      % Spatial regularization identity like
lambda_t = 0.01 * lambda;                     % Temporal regularization (adjust as needed)

% Construct temporal difference matrix L_t (T-1 × T)
L_t = diag(-ones(n_timepoints,1), 0) + diag(ones(n_timepoints-1,1), 1);
L_t = L_t(1:n_timepoints-1, :);               % Remove last row (optional)

% Compute terms for the linear system: (A'*A + λI + λ_t L_t'*L_t) x = A'*B
ATA = A' * A;                                  % [3Nsources × 3Nsources]
ATB = A' * B;                                  % [3Nsources × T]
LtLtT = L_t' * L_t;                            % [T × T] (temporal smoothing operator)

% Reshape ATB into a vector for CG solver
b_vec = ATB(:);                                % [3Nsources*T × 1]

% Define the linear operator for CG (Ax = (ATA ⊗ I_T + λ I + λ_t I_N ⊗ LtLtT) x)
% We'll use a function handle to avoid explicitly constructing the large matrix
linear_operator = @(x) reshape(...
    ATA * reshape(x, [3*n_sources, n_timepoints]) + ...          % ATA * X
    lambda * reshape(x, [3*n_sources, n_timepoints]) + ...       % λ * X
    lambda_t * reshape(x, [3*n_sources, n_timepoints]) * LtLtT, ... % λ_t * X * LtLtT
    [3*n_sources * n_timepoints, 1]);

% Solve using conjugate gradient (preconditioned with diagonal of ATA + λI)
% M = diag(kron(ones(n_timepoints,1), diag(ATA)) + lambda * eye(3*n_sources*n_timepoints)
% Instead of constructing M, compute M\x implicitly:
preconditioner = @(x) reshape(...
    (1./(diag(ATA) + lambda)) .* reshape(x, [3*n_sources, n_timepoints]), ...  % (ATA + λI)\x
    [3*n_sources * n_timepoints, 1]);

% Solve using PCG with implicit preconditioner
x_vec = pcg(linear_operator, b_vec, 1e-6, 500, preconditioner);  % Use function handle

% Reshape solution back to [3Nsources × T]
x_est = reshape(x_vec, [3*n_sources, n_timepoints]);

% Reshape into source space [Nsources × 3 × T]
source_activity = zeros(n_sources, 3, n_timepoints);
for i = 1:n_sources
    source_activity(i,:,:) = x_est((3*i-2):3*i, :);
end

% Compute power across orientations
J_vec = squeeze(sum(source_activity.^2, 2));  % [Nsources × T]


%% ====================== 7. Visualize Time-Evolving Top Dipoles ======================
figure;

% Create scalp surface interpolation
S = scatteredInterpolant(elec.elecpos(:,1), elec.elecpos(:,2), elec.elecpos(:,3), 'natural');
[minX, maxX] = bounds(elec.elecpos(:,1));
[minY, maxY] = bounds(elec.elecpos(:,2));
padding = 1; 
[xx, yy] = meshgrid(linspace(minX-padding, maxX+padding, 20), linspace(minY-padding, maxY+padding, 20));
zz = S(xx, yy);

% Plot scalp surface
h_surf = surf(xx, yy, zz, 'FaceColor', [0.9 0.9 0.9], 'EdgeColor', 'none', 'FaceAlpha', 0.4);
hold on;

% Plot all dipoles
scatter3(sourcemodel.pos(:,1), sourcemodel.pos(:,2), sourcemodel.pos(:,3), ...
    5, 'k', 'filled', 'MarkerFaceAlpha', 0.3);

% Set viewing properties
axis equal tight;
grid on;
view(135, 30);
zlim([-150 150]);

% Prepare video
videoFile = 'top_dipoles_evolution.mp4';
v = VideoWriter(videoFile, 'MPEG-4');
v.FrameRate = 7;
v.Quality = 100;
open(v);

% Get time points (use every 5th sample for efficiency)
time_points = 1:5:size(J_vec, 2);
T = length(time_points);

% Parameters
k = 10; % Number of top dipoles to highlight
active_dipoles_prev = [];
inside_idx = find(sourcemodel.inside); % This gives indices of inside dipoles in sourcemodel.pos

for t_idx = 1:T
    t = time_points(t_idx);
    
    % Get current activity (from J_vec)
    current_activity = J_vec(:, t);
    
    % Get top k dipoles (only consider inside dipoles)
    [~, sorted_indices] = sort(current_activity, 'descend'); % No need to index with sourcemodel.inside
    %inside_idx = find(sourcemodel.inside);
    top_k_dipoles = inside_idx(sorted_indices(1:min(k, length(sorted_indices))));
    
    % Reset previous actives
    if ~isempty(active_dipoles_prev)
        scatter3(sourcemodel.pos(active_dipoles_prev, 1), ...
                sourcemodel.pos(active_dipoles_prev, 2), ...
                sourcemodel.pos(active_dipoles_prev, 3), 5, 'k', 'filled', 'MarkerFaceAlpha', 0.3);
    end
    
    % Highlight current top-k dipoles
    activity_norm = current_activity(top_k_dipoles)/max(current_activity);
    marker_sizes = 50 + 100*activity_norm;
    
    h = scatter3(sourcemodel.pos(top_k_dipoles, 1), ...
                sourcemodel.pos(top_k_dipoles, 2), ...
                sourcemodel.pos(top_k_dipoles, 3), ...
                marker_sizes, 'r', 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1);
    
    % Add time annotation
    time_sec = data.time{1}(t);
    time_text = sprintf('Time: %.3f s', time_sec);
    if exist('htext', 'var'), delete(htext); end
    htext = text(minX-padding+5, minY-padding+5, max(zz(:))+10, time_text, ...
               'FontSize', 12, 'Color', 'k', 'BackgroundColor', 'w');
    
    % Update previous actives
    active_dipoles_prev = top_k_dipoles;
    
    % Capture frame
    frame = getframe(gcf);
    writeVideo(v, frame);
    pause(0.01);
    delete(h);
end

close(v);
disp(['Saved dipole evolution video as: ' videoFile]);

%% ====================== 8. EEG Reconstruction Comparison ======================
time_window = 1; % seconds
n_samples = min(round(time_window * data.fsample), size(data.trial{1}, 2));

% Get inside dipole indices
inside_idx = find(sourcemodel.inside);
n_dipoles = length(inside_idx);

%{ 
%Build leadfield matrix A (Nchannels × 3Ndipoles)
A = zeros(length(data.label), 3*n_dipoles);
for i = 1:n_dipoles
    idx = inside_idx(i);
    A(:, (3*i-2):3*i) = sourcemodel.leadfield{idx};
end
%}

% Reconstruct EEG: A * J_vec (where J_vec is [3Ndipoles × T])
% For the selected time window:
est = A * x_est(:, 1:n_samples); % x_est is [3Ndipoles × T] from Part 6

% Plot comparison
figure;
subplot(2,1,1);
plot(data.time{1}(1:n_samples), data.trial{1}(:, 1:n_samples));
title('Original EEG');
ylabel('Amplitude (\muV)');

subplot(2,1,2);
plot(data.time{1}(1:n_samples), est);
title('Reconstructed EEG');
xlabel('Time (s)');
ylabel('Amplitude (\muV)');

% Now plot one for each channel to see how close it is to the original, in a single figure
figure;
n_channels = length(data.label); % Number of channels (should be 21)

for ch = 1:n_channels
    subplot(n_channels, 1, ch);
    hold on;
    
    % Plot original EEG (real)
    plot(data.time{1}(1:n_samples), data.trial{1}(ch, 1:n_samples), 'b');
    
    % Plot reconstructed EEG
    plot(data.time{1}(1:n_samples), est(ch, 1:n_samples), 'r--');
    
    % Add channel label
    ylabel(data.label{ch});
    
    % Only show x-axis label for bottom plot
    if ch ~= n_channels
        set(gca, 'XTickLabel', []);
    else
        xlabel('Time (s)');
    end
    
    % Add legend for first subplot only (to save space)
    if ch == 1
        legend('Original', 'Reconstructed', 'Location', 'best');
    end
end

% Adjust figure properties for better visualization
%set(gcf, 'Position', [100, 100, 800, 1200]); % Make figure taller
sgtitle('EEG Channel Comparison: Original vs Reconstructed');

% Calculate reconstruction error
recon_error = data.trial{1}(:, 1:n_samples) - est;
recon_quality = norm(recon_error, 'fro') / norm(data.trial{1}(:, 1:n_samples), 'fro');
disp(['Reconstruction error (Frobenius norm): ', num2str(recon_quality)]);