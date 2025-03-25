function L_s = construct_spatial_laplacian(dipole_positions, sigma, neighborhood_threshold)
% Constructs the spatial Laplacian matrix L_s = D - W
    % where W is the weight matrix based on dipole distances,
    % and D is the degree matrix
	% as we did with Braunstein
    
    % Inputs:
    %   dipole_positions: N_dipoles x 3 matrix of dipole positions (N_dipoles, 3 coordinates)
    %   sigma: Scaling parameter for the weight matrix
    
    % Number of dipoles
    N_dipoles = size(dipole_positions, 1);

    % Compute pairwise distances between dipoles
    distances = pdist2(dipole_positions, dipole_positions);
    
    % Apply neighborhood threshold: set weights to zero for dipoles beyond the threshold
    W = exp(-distances.^2 / sigma^2); % Compute weights
    W(distances > neighborhood_threshold) = 0; % Set weights to zero for distant dipoles
    
    % Ensure W is symmetric (optional, but good practice)
    W = (W + W') / 2;
    
    % Compute the degree matrix D (N_dipoles x N_dipoles)
    D = diag(sum(W, 2));
    
    % Construct the spatial Laplacian for dipoles (N_dipoles x N_dipoles)
    L_dipoles = D - W;
    
    % Expand L_dipoles to account for 3 orientations per dipole
    % L_s will be a block-diagonal matrix with L_dipoles repeated for each orientation
    L_s = kron(L_dipoles, speye(3)); % N_sources x N_sources, where N_sources = 3 * N_dipoles
end