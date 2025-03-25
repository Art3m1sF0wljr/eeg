function [B, Xsimulated] = professor_generated_B(A, good_dipoles, T, active_sources, space_smooth_factor)
    % Generates EEG data B using the forward problem and simulated source activity.
    % This version creates smoothly varying activation patterns for the dipoles.
    %
    % Inputs:
    %   A: Leadfield matrix (Nch x Nsources)
    %   good_dipoles: Indices of active dipoles (e.g., DipoleField.inside)
    %   T: Number of time points
    %   active_sources: Number of active sources
    %   space_smooth_factor: Spatial smoothness factor
    %
    % Outputs:
    %   B: Simulated EEG data (Nch x T)
    %   Xsimulated: Simulated source activity (Nsources x T)

    % Number of sources
    Nsources = size(A, 2);

    % Initialize source activity matrix
    Xsimulated = zeros(Nsources, T);

    % Define time discretization parameters
    fs = 64; % Sampling frequency (Hz) idk why, he chose it
    tt = linspace(-0.5, 0.5, T); % Time vector with T points
    ss = space_smooth_factor; % Standard deviation for the Gaussian 
    g = diff(exp(-tt.^2 / 2 / ss^2)); % Derivative of a Gaussian

    % Ensure g is a column vector for concatenation
    g = g(:); % Convert g to a column vector

    % Randomly select active sources
    active_source_indices = randperm(length(good_dipoles), active_sources); % Randomly select active dipoles
    active_source_indices = good_dipoles(active_source_indices); % Map to global source indices

    % Simulate smoothly varying activation patterns for the active sources
    for i = 1:length(active_source_indices)
        % Create a window function with random onset
        nn = 3; % Number of active time points
        vv = [ones(nn, 1); zeros(T - nn, 1)]; % Window function with T points
        vv = vv(randperm(T)); % Randomly permute the window function

        % Convolve with Gaussian derivative kernel to create smooth activation
        activation_pattern = conv(vv, [g; 0], 'same'); % Ensure same length as T

        % Scale by a random activation value
        activation_strength = rand(1); % Random activation strength
        Xsimulated(active_source_indices(i), :) = activation_strength * activation_pattern;
    end

    % Simulate EEG data over T time points
    B = A * Xsimulated; % Forward EEG computation (Nch x T)
end