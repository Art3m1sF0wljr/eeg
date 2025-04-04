close all; clear all; clc;
load('DipoleField');
%load('elec.mat'); % Make sure this contains electrode positions

%% ====================== 1. Load and Preprocess EEG Data ======================
% Load EEG data
[fname, fdir] = uigetfile('*.edf', 'Select the EEG file');
[hdr, record] = edfread([fdir fname]);
% Select only the first second of data
samples_to_keep = hdr.samples(1); % Number of samples in 1 second
record = record(:, 1:min(samples_to_keep, size(record,2))); % Take first second or all if shorter   
% Convert to proper FieldTrip structure
data = [];
data.trial = {record};
data.time = {(0:size(record,2)-1)/hdr.samples(1)};
data.label = hdr.label;
data.fsample = hdr.samples(1);

% Simple preprocessing (bandpass filter 1-30 Hz)
cfg = [];
cfg.bpfilter = 'yes';
cfg.bpfreq = [1 30];
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

%% ====================== 5. Prepare Timelock Data PROPERLY ======================
cfg = [];
cfg.covariance = 'yes';
cfg.covariancewindow = 'all'; % Compute covariance over entire data
cfg.keeptrials = 'no'; % Average over trials
timelock = ft_timelockanalysis(cfg, data);

% Verify timelock structure
if ~isfield(timelock, 'cov')
    error('Timelock data does not contain covariance matrix. Check your data and preprocessing.');
end
%% ====================== 6. Robust Chunked sLORETA Processing ======================
cfg = [];
cfg.method = 'sloreta';
cfg.headmodel = headmodel;
cfg.sourcemodel = sourcemodel; % Use the sourcemodel parameter instead of passing as input
cfg.lambda = '10%';
cfg.keepfilter = 'yes';
cfg.feedback = 'text';
cfg.projectnoise = 'yes'; % Important for sLORETA
cfg.keepleadfield = 'yes';

% Initialize output structure
source = struct();
source.pos = sourcemodel.pos;
source.inside = sourcemodel.inside;
source.avg.pow = zeros(size(sourcemodel.pos,1), 1);
source.avg.filter = cell(size(sourcemodel.pos,1), 1);

% Get all inside dipole indices
inside_idx = find(sourcemodel.inside);
n_dipoles = length(inside_idx);
chunk_size = 3000;  % Adjust based on your system memory
n_chunks = ceil(n_dipoles/chunk_size);

for chunk = 1:n_chunks
    fprintf('Processing chunk %d/%d (%.1f%%)...\n', ...
            chunk, n_chunks, 100*chunk/n_chunks);
    
    % Get current chunk indices
    chunk_start = (chunk-1)*chunk_size + 1;
    chunk_end = min(chunk*chunk_size, n_dipoles);
    current_idx = inside_idx(chunk_start:chunk_end);
    
    % Create temporary sourcemodel with only current dipoles
    temp_sourcemodel = sourcemodel;
    temp_sourcemodel.inside = false(size(sourcemodel.inside));
    temp_sourcemodel.inside(current_idx) = true;
    
    % Update cfg with temporary sourcemodel
    cfg.sourcemodel = temp_sourcemodel;
    
    % Process current chunk
    temp_source = ft_sourceanalysis(cfg, timelock);
    
    % Store results
    source.avg.pow(current_idx) = temp_source.avg.pow(temp_source.inside);
    source.avg.filter(current_idx) = temp_source.avg.filter(temp_source.inside);
    
    clear temp_source temp_sourcemodel  % Free memory
end

%% ====================== 7. Visualize Time-Evolving Top Dipoles ======================
%{
% Create scalp surface interpolation
figure;
S = scatteredInterpolant(elec.elecpos(:,1), elec.elecpos(:,2), elec.elecpos(:,3), 'natural');
[minX, maxX] = bounds(elec.elecpos(:,1));
[minY, maxY] = bounds(elec.elecpos(:,2));
[xx, yy] = meshgrid(linspace(minX, maxX, 50), linspace(minY, maxY, 50));
zz = S(xx, yy);

% Plot every 5th dipole for better visibility
plot_indices = 1:5:size(sourcemodel.pos,1);

% Plot dipole activity
subplot(1,2,1);
surf(xx, yy, zz, 'FaceColor', [.8 .8 .8], 'EdgeColor', 'none', 'FaceAlpha', 0.3);
hold on;
scatter3(sourcemodel.pos(plot_indices,1), sourcemodel.pos(plot_indices,2), sourcemodel.pos(plot_indices,3), ...
    30, source.avg.pow(plot_indices), 'filled');
colorbar; title('Dipole Activity'); axis equal;

% Plot top 10 dipoles
[~, sorted_idx] = sort(source.avg.pow, 'descend');
top_dipoles = sorted_idx(1:min(10,length(sorted_idx)));

subplot(1,2,2);
scatter3(sourcemodel.pos(plot_indices,1), sourcemodel.pos(plot_indices,2), sourcemodel.pos(plot_indices,3), 10, 'k');
hold on;
scatter3(sourcemodel.pos(top_dipoles,1), sourcemodel.pos(top_dipoles,2), sourcemodel.pos(top_dipoles,3), ...
    120, source.avg.pow(top_dipoles), 'filled', 'MarkerEdgeColor', 'k');
title('Top 10 Strongest Dipoles'); axis equal; colorbar;
%}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% First create a static visualization of all dipoles with scalp surface
%% ====================== 7. Visualize Time-Evolving Top Dipoles (Improved Layout) ======================
% First create a static visualization of all dipoles with scalp surface
figure; % Wider figure window

% Create a larger scalp surface interpolation
S = scatteredInterpolant(elec.elecpos(:,1), elec.elecpos(:,2), elec.elecpos(:,3), 'natural');
[minX, maxX] = bounds(elec.elecpos(:,1));
[minY, maxY] = bounds(elec.elecpos(:,2));
padding = 1; % Add padding around electrodes
[xx, yy] = meshgrid(linspace(minX-padding, maxX+padding, 20), linspace(minY-padding, maxY+padding, 20));
zz = S(xx, yy);

% Create a subplot with better proportions
%subplot(1,1,1); % Use full figure area

% Plot scalp surface with better visibility
h_surf = surf(xx, yy, zz, 'FaceColor', [0.9 0.9 0.9], 'EdgeColor', 'none', 'FaceAlpha', 0.4);
hold on;

% Plot all dipoles with better sizing
scatter3(sourcemodel.pos(:,1), sourcemodel.pos(:,2), sourcemodel.pos(:,3), ...
    5, 'k', 'filled', 'MarkerFaceAlpha', 0.3);

% Set better axis properties
%axis equal tight;
grid on;
view(135, 30); % Better viewing angle
%set(gca, 'FontSize', 12, 'LineWidth', 1.5);
%xlabel('X (mm)', 'FontSize', 14);
%ylabel('Y (mm)', 'FontSize', 14);
%zlabel('Z (mm)', 'FontSize', 14);
%title('Top 10 Active Dipoles Over Time', 'FontSize', 16);

% Adjust lighting for better visibility
%light('Position',[1 1 1],'Style','infinite');
%lighting gouraud;
%material dull;
%h_surf.FaceLighting = 'gouraud';
zlim([-150 150]); % Adjust Z-range to match X & Y
view(135, 45); % Increase elevation angle

% Prepare video with better resolution
videoFile = 'top10_dipoles_evolution_improved.mp4';
v = VideoWriter(videoFile, 'MPEG-4');
v.FrameRate = 20;
v.Quality = 100; % Higher quality
open(v);

% Get time information
n_samples = length(data.time{1});
time_points = 1:2:n_samples;  % Analyze every 10th sample for efficiency
T = length(time_points);

% Parameters
k = 10; % Number of top dipoles to highlight
active_dipoles_prev = [];

% Get all inside dipole indices
inside_idx = find(sourcemodel.inside);

for t_idx = 1:T
    t = time_points(t_idx);
    
    % Get current time window data
    time_window = max(1,t-5):min(n_samples,t+5);
    current_data = data.trial{1}(:, time_window);
    
    % Compute source activity
    source_activity = zeros(length(inside_idx), 1);
    for i = 1:length(inside_idx)
        dipole_idx = inside_idx(i);
        source_activity(i) = norm(source.avg.filter{dipole_idx} * current_data, 'fro');
    end
    
    % Get top k dipoles
    [~, sorted_indices] = sort(source_activity, 'descend');
    top_k_dipoles = inside_idx(sorted_indices(1:min(k, length(sorted_indices))));
    
    % Reset previous actives
    if ~isempty(active_dipoles_prev)
        scatter3(sourcemodel.pos(active_dipoles_prev, 1), ...
                sourcemodel.pos(active_dipoles_prev, 2), ...
                sourcemodel.pos(active_dipoles_prev, 3), 5, 'k', 'filled', 'MarkerFaceAlpha', 0.3);
    end
    
    % Highlight current top-k dipoles with size proportional to activity
    activity_norm = source_activity(sorted_indices(1:k))/max(source_activity);
    marker_sizes = 50 + 100*activity_norm; % Vary size from 50 to 150
    
    h = scatter3(sourcemodel.pos(top_k_dipoles, 1), ...
                sourcemodel.pos(top_k_dipoles, 2), ...
                sourcemodel.pos(top_k_dipoles, 3), ...
                marker_sizes, 'r', 'filled', ...
                'MarkerEdgeColor', 'k', 'LineWidth', 1);
    
    % Add time annotation
    time_sec = data.time{1}(t);
    time_text = sprintf('Time: %.3f s', time_sec);
    if exist('htext', 'var'), delete(htext); end
    htext = text(minX-padding+5, minY-padding+5, max(zz(:))+10, time_text, ...
               'FontSize', 5, 'Color', 'k', 'BackgroundColor', 'w');
    
    % Update previous actives
    active_dipoles_prev = top_k_dipoles;
    
    % Capture frame
    frame = getframe(gcf);
    writeVideo(v, frame);
    
    % Pause for visualization
    pause(0.0001);
    
    % Delete the current highlight for next frame
    delete(h);
end

hold off;
close(v);
disp(['Saved improved dipole evolution video as: ' videoFile]);
%% ====================== 8. EEG Reconstruction Comparison ======================
% Reconstruct EEG from available channels
time_window = 1; % seconds
n_samples = min(round(time_window * data.fsample), size(data.trial{1},2));

% Get indices of inside dipoles
inside_dipoles = find(source.inside);
n_dipoles = length(inside_dipoles);

% Use the valid channels that work with both leadfield and filter
valid_ch = [];
for ch = [4, 6, 19, 21] % Try the channels that previously worked
    valid = true;
    for i = 1:n_dipoles
        dipole_idx = inside_dipoles(i);
        % Check if channel exists in both leadfield and filter
        if ch > size(sourcemodel.leadfield{dipole_idx}, 1) || ...
           ch > size(source.avg.filter{dipole_idx}, 2)
            valid = false;
            break;
        end
    end
    if valid
        valid_ch = [valid_ch, ch];
    end
end

if isempty(valid_ch)
    error('No valid channels found that work with all dipoles');
end

n_channels = length(valid_ch);
disp(['Using channels: ' num2str(valid_ch)]);

% Build leadfield matrix (n_channels × 3n_dipoles)
A = zeros(n_channels, 3*n_dipoles);
for i = 1:n_dipoles
    dipole_idx = inside_dipoles(i);
    A(:, (i-1)*3+1:i*3) = sourcemodel.leadfield{dipole_idx}(valid_ch, :);
end

% Build filter matrix (3n_dipoles × n_channels)
w = zeros(3*n_dipoles, n_channels);
for i = 1:n_dipoles
    dipole_idx = inside_dipoles(i);
    w((i-1)*3+1:i*3, :) = source.avg.filter{dipole_idx}(:, valid_ch);
end

% Verify dimensions
disp(['Leadfield (A) dimensions: ' num2str(size(A))]);
disp(['Filter (w) dimensions: ' num2str(size(w))]);
disp(['EEG data dimensions: ' num2str(size(data.trial{1}(valid_ch, 1:n_samples)))]);

% Reconstruct EEG: A * w * original_data
est = A * w * data.trial{1}(valid_ch, 1:n_samples);

% Plot comparison
figure;
subplot(2,1,1);
plot(data.time{1}(1:n_samples), data.trial{1}(valid_ch, 1:n_samples));
title(['Original EEG (' num2str(n_channels) ' Channels)']);
ylabel('Amplitude (\muV)');
legend(data.label(valid_ch));

subplot(2,1,2);
plot(data.time{1}(1:n_samples), est);
title(['Reconstructed EEG (' num2str(n_channels) ' Channels)']);
xlabel('Time (s)');
ylabel('Amplitude (\muV)');
legend(strcat('Recon-', data.label(valid_ch)));

% Calculate reconstruction error
recon_error = data.trial{1}(valid_ch, 1:n_samples) - est;
recon_quality = norm(recon_error, 'fro')/norm(data.trial{1}(valid_ch, 1:n_samples), 'fro');
disp(['Reconstruction quality metric (0=perfect): ' num2str(recon_quality)]);