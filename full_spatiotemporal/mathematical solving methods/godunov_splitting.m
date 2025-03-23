%solve a function of the kind d(f(x,t))/dt=div(A grad(f(x,t)))
clear all;close all
function godunov_diffusion_solver()
    % Parameters
    L = 1;          % Length of domain
    Nx = 50;        % Number of spatial points
    dx = L / Nx;    % Spatial step
    T = 0.1;        % Final time
    dt = 0.001;     % Time step
    Nt = T / dt;    % Number of time steps
    
    x = linspace(0, L, Nx)';  % Spatial grid
    f = exp(-100 * (x - 0.5).^2);  % Initial condition (Gaussian)
    A = 1;          % Diffusion coefficient (can be a function of x)
    
    % Constructing the diffusion matrix (assuming constant A)
    e = ones(Nx,1);
    D = spdiags([e -2*e e], [-1 0 1], Nx, Nx) / dx^2;
    
    % Apply Neumann boundary conditions (zero gradient)
    D(1,1) = -1/dx^2; D(1,2) = 1/dx^2;
    D(end,end-1) = 1/dx^2; D(end,end) = -1/dx^2;
    
    % Time stepping using Godunov splitting
    for n = 1:Nt
        % Step 1: Solve the diffusion equation (implicit Euler)
        f = (speye(Nx) - dt * A * D) \ f;
        
        % Additional steps for other terms can be inserted here
        
        % Visualization
        if mod(n, 50) == 0
            plot(x, f, 'b', 'LineWidth', 2);
            title(['Time = ', num2str(n * dt)]);
            xlabel('x'); ylabel('f(x,t)');
            drawnow;
        end
    end
end

godunov_diffusion_solver();