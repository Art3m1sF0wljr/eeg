clear all; close all;

%% Load DipoleField and Initialize Variables
load('DipoleField');
Nsources = 3 * sum(DipoleField.inside);
Nch = length(DipoleField.label); % Number of EEG channels

%% Create Interpolated Scalp Surface
S = scatteredInterpolant(elec.elecpos(:,1),elec.elecpos(:,2),elec.elecpos(:,3),'natural');
minX = min(elec.elecpos(:,1)); maxX = max(elec.elecpos(:,1));
minY = min(elec.elecpos(:,2)); maxY = max(elec.elecpos(:,2));
[xx, yy] = meshgrid(linspace(minX, maxX, 20), linspace(minY, maxY, 20));
zz = S(xx, yy);

% Remove points outside the head convex hull
tess = convhulln(elec.elecpos(:,1:2));
in = inhull([xx(:) yy(:)], elec.elecpos(:,1:2), tess);
zz(~in) = nan;

good_dipoles = find(DipoleField.inside == 1); % Select dipoles inside the brain
A = zeros(Nch, Nsources); % Leadfield matrix

%% Construct Leadfield Matrix
n = 1;
for i = 1:Nsources/3
    A(:, n:n+2) = DipoleField.leadfield{good_dipoles(i)};
    n = n + 3;
end

%% Simulate EEG Signals from Sparse Source Activation
active_sources = 5;
Xsimulated = [rand(active_sources, 1); zeros(Nsources - active_sources, 1)];
AS = randperm(Nsources);
Xsimulated = Xsimulated(AS);
b = A * Xsimulated; % EEG measurement at electrodes

%% Plot the Forward Model (EEG scalp potentials)
F = scatteredInterpolant(elec.elecpos(:,1), elec.elecpos(:,2), elec.elecpos(:,3), b, 'natural');
surf(xx, yy, zz, F(xx, yy, zz), 'EdgeColor', 'none');
alpha(.6); xlabel('x'); ylabel('y'); zlabel('z'); axis equal; hold on;

% Highlight Active Sources
qq = find(Xsimulated > 0);
for Num = 1:length(qq)
    meanA = Xsimulated(qq(Num));
    Num_dipole = round(qq(Num) / 3 + 1);
    color_level = round(100 * (.9 - .9 * meanA / max(Xsimulated))) / 100;
    plot3(DipoleField.pos(good_dipoles(Num_dipole),1), DipoleField.pos(good_dipoles(Num_dipole),2), ...
          DipoleField.pos(good_dipoles(Num_dipole),3), 'o', 'color', color_level * [1 1 1], ...
          'MarkerSize', 1 + 5 * meanA / max(Xsimulated), 'LineWidth', 1 + 2 * meanA / max(Xsimulated));
end

%% Spatiotemporal Kalman Filtering Solution
T = size(b, 2); % Number of time steps
J_est = zeros(Nsources, T); % Estimated source activity
P = eye(Nsources) * 1e6; % Initial state covariance

% Define System Noise and Measurement Noise
sigma_eta = 1e-3; C_eta = sigma_eta^2 * eye(Nsources);
sigma_eps = 1e-2; C_eps = sigma_eps^2 * eye(Nch);

% Define State Transition Matrix (AR model with neighborhood interaction)
a1 = 0.9; b1 = 0.1;
A_state = a1 * eye(Nsources) + b1 * (rand(Nsources) < 0.02); % Sparse connectivity

for t = 2:T
    %% Prediction Step
    J_pred = A_state * J_est(:, t-1);
    P_pred = A_state * P * A_state' + C_eta;
    
    %% Compute Kalman Gain
    S = A * P_pred * A' + C_eps;
    K_gain = P_pred * A' / S;
    
    %% Update Step
    J_est(:, t) = J_pred + K_gain * (b(:, t) - A * J_pred);
    P = (eye(Nsources) - K_gain * A) * P_pred;
end

%% Plot the Estimated Sources
figure;
subplot(2,1,1);
plot(Xsimulated, 'k'); hold on;
plot(J_est(:, round(T/5)), 'r');
plot(J_est(:, round(T/2)), 'm');
plot(J_est(:, T), 'g');
legend('True Sources', 'Estimation (T/5)', 'Estimation (T/2)', 'Final Estimation');

title('Kalman Filtered Source Estimation');
subplot(2,1,2);
plot(sqrt(mean(J_est.^2, 2)), 'b');
title('Estimated Source Energy over Time');
