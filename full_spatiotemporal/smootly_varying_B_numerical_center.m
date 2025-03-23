function [B, Xsimulated] = smootly_varying_B_numerical_center(A, good_dipoles, T, active_sources, neighborhood_size, base_probability, space_smooth_factor)
    % Generates EEG data B using the forward problem and simulated source activity.
    %
    % Inputs:
    %   A: Leadfield matrix (Nch x Nsources)
    %   good_dipoles: Indices of dipoles inside the brain (N x 1)
    %   T: Number of time points
    %   active_sources: Number of active sources at each time point
    %   neighborhood_size: Size of the neighborhood for source activation
    %   base_probability: Base probability of a source lighting up
    %   space_smooth_factor: Spatial smoothness factor for source activity
    %
    % Outputs:
    %   B: Simulated EEG data (Nch x T)
    %   Xsimulated: Simulated source activity (Nsources x T)

    % Number of sources
    Nsources = size(A, 2);

    % Initialize source activity matrix
    Xsimulated = zeros(Nsources, T);

    % Gaussian envelope for smooth temporal variation
    time_smooth = exp(-linspace(-2, 2, T).^2);

    % Initialize active sources at the first time step
    active_indices = randperm(length(good_dipoles), active_sources); % Randomly select active sources
    Xsimulated(active_indices, 1) = rand(active_sources, 1) .* time_smooth(1); % Apply initial activation

    % Simulate source activity over time with smooth variations
    for t = 2:T
        % Find all candidate sources within the neighborhood of currently active sources
        candidate_indices = [];
        for i = 1:length(active_indices)
            % Define the neighborhood around the current active source
            lower_bound = max(1, active_indices(i) - neighborhood_size);
            upper_bound = min(length(good_dipoles), active_indices(i) + neighborhood_size);
            candidate_indices = [candidate_indices, lower_bound:upper_bound];
        end
        candidate_indices = unique(candidate_indices); % Remove duplicates

        % Exclude currently active sources from candidates (they can turn off but not light up again)
        candidate_indices = setdiff(candidate_indices, active_indices);

        % Calculate probabilities of lighting up based on distance from active sources
        probabilities = zeros(length(candidate_indices), 1);
        for i = 1:length(candidate_indices)
            % Find the minimum distance (in numbering) to any active source
            min_distance = min(abs(candidate_indices(i) - active_indices));
            % Probability is inversely proportional to distance
            probabilities(i) = base_probability / (1 + min_distance);
        end

        % Normalize probabilities to ensure they sum to 1
        probabilities = probabilities / sum(probabilities);

        % Select new active sources based on the calculated probabilities
        num_new_sources = active_sources; % Number of new sources to select
        new_active_indices = randsample(candidate_indices, num_new_sources, true, probabilities);

        % Random activation for new active sources
        temp_activity = rand(num_new_sources, 1);
        % Apply temporal smoothing
        Xsimulated(new_active_indices, t) = temp_activity .* time_smooth(t);

        % Handle turning off previously active sources
        turn_off_probability = 1 / 6; % Probability of turning off
        turn_off_mask = rand(length(active_indices), 1) < turn_off_probability;
        active_indices(turn_off_mask) = []; % Remove sources that turn off

        % Combine new and remaining active sources
        active_indices = unique([active_indices(:); new_active_indices(:)]);

        % Ensure exactly 6 active sources (add or remove if necessary)
        if length(active_indices) > active_sources
            active_indices = active_indices(1:active_sources); % Keep only the first 6
        elseif length(active_indices) < active_sources
            % Add additional sources randomly from candidates
            additional_indices = randsample(candidate_indices, active_sources - length(active_indices), true, probabilities);
            active_indices = [active_indices; additional_indices];
        end
    end

    % Spatial smoothness: smooth source activity over space
    Xsimulated = smooth_source_activity(Xsimulated, space_smooth_factor); % Apply spatial smoothing

    % Simulate EEG data over T time points
    B = A * Xsimulated; % Forward EEG computation (Nch x T)
end

% Helper function for smoothing source activity over space
function smoothed_activity = smooth_source_activity(activity, factor)
    smoothed_activity = activity;
    for t = 1:size(activity, 2)
        smoothed_activity(:, t) = smoothdata(activity(:, t), 'movmean', factor);
    end
end