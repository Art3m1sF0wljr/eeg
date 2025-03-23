function strang_splitting_diffusion()
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
    
    % Construct the diffusion matrix for half-time steps
    D_half = construct_diffusion_matrix(A_half, dx, Nx, 0.5 * dt);
    D_full = construct_diffusion_matrix(A_half, dx, Nx, dt);
    
    % Time stepping using Strang splitting
    for n = 1:Nt
        % Step 1: Half-step evolution
        f = (speye(Nx) - D_half) \ f;
        
        % Step 2: Full-step evolution
        f = (speye(Nx) - D_full) \ f;
        
        % Step 3: Another half-step evolution
        f = (speye(Nx) - D_half) \ f;
        
        % Visualization
        if mod(n, 50) == 0
            plot(x, f, 'b', 'LineWidth', 2);
            title(['Time = ', num2str(n * dt)]);
            xlabel('x'); ylabel('f(x,t)');
            drawnow;
        end
    end
end

function D = construct_diffusion_matrix(A_half, dx, Nx, dt)
    % Constructs the diffusion matrix for a given timestep dt
    D = spalloc(Nx, Nx, 3 * Nx);  % Sparse matrix
    
    for i = 2:Nx-1
        D(i, i-1) = dt * A_half(i-1) / dx^2;
        D(i, i)   = -dt * (A_half(i-1) + A_half(i)) / dx^2;
        D(i, i+1) = dt * A_half(i) / dx^2;
    end
    
    % Neumann boundary conditions (zero flux)
    D(1,1) = -dt * A_half(1) / dx^2;
    D(1,2) = dt * A_half(1) / dx^2;
    D(end,end-1) = dt * A_half(end) / dx^2;
    D(end,end) = -dt * A_half(end) / dx^2;
end

strang_splitting_diffusion()