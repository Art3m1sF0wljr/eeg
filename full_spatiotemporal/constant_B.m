function [B, Xsimulated] = constant_B(A, active_source_indices, activation_values, T)
    % Generates EEG data B using the forward problem and simulated source activity.
    % This version keeps a predefined set of sources active with constant activation values.
    %
    % Inputs:
    %   A: Leadfield matrix (Nch x Nsources)
    %   active_source_indices: Indices of the active sources (e.g., [1, 2, 3, ..., active_sources])
    %   activation_values: Activation values for the active sources (active_sources x 1)
    %   T: Number of time points
    %
    % Outputs:
    %   B: Simulated EEG data (Nch x T)
    %   Xsimulated: Simulated source activity (Nsources x T)

    % Number of sources
    Nsources = size(A, 2);

    % Initialize source activity matrix
    Xsimulated = zeros(Nsources, T);

    % Assign constant activation to the predefined sources over all time points
    for t = 1:T
        Xsimulated(active_source_indices, t) = activation_values; % Keep the same activation over time
    end

    % Simulate EEG data over T time points
    B = A * Xsimulated; % Forward EEG computation (Nch x T)
end