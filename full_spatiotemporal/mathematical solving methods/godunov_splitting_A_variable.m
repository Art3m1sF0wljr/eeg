%solve a function of the kind d(f(x,t))/dt=div(A(x) grad(f(x,t)))
clear all;close all
function godunov_diffusion_solver_variable_A()
    % Parameters
    L = 1;          % Length of domain
    Nx = 50;        % Number of spatial points
    dx = L / Nx;    % Spatial step
    T = 0.1;        % Final time
    dt = 0.001;     % Time step
    Nt = T / dt;    % Number of time steps
    
    x = linspace(0, L, Nx)';  % Spatial grid
    f = exp(-100 * (x - 0.5).^2);  % Initial condition (Gaussian)
    
    % Variable diffusion coefficient A(x)
    A = 1 + 0.5 * sin(2 * pi * x);  % Example: oscillating A(x)
    
    % Compute A at half points (cell interfaces)
    A_half = zeros(Nx-1,1);
    for i = 1:Nx-1
        A_half(i) = (A(i) + A(i+1)) / 2;  % Arithmetic mean
    end
    
    % Construct the diffusion matrix with variable A(x)
    D = spalloc(Nx, Nx, 3 * Nx);  % Sparse matrix
    
    for i = 2:Nx-1
        D(i, i-1) = A_half(i-1) / dx^2;
        D(i, i) = -(A_half(i-1) + A_half(i)) / dx^2;
        D(i, i+1) = A_half(i) / dx^2;
    end
    
    % Neumann boundary conditions (zero flux)
    D(1,1) = -A_half(1) / dx^2;
    D(1,2) = A_half(1) / dx^2;
    D(end,end-1) = A_half(end) / dx^2;
    D(end,end) = -A_half(end) / dx^2;
    
    % Time stepping using Godunov splitting
    for n = 1:Nt
        % Step 1: Solve the diffusion equation (implicit Euler)
        f = (speye(Nx) - dt * D) \ f;
        
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
godunov_diffusion_solver_variable_A();