% Copyright (c) 2023 Matteo Giordano
%
% Codes accompanying the article "Bayesian nonparametric inference in 
% PDE models: asymptotic theory and implementation" by Matteo Giordano

%%
% Let O be the disk (in R^2) with radius r. Consider the 2D elliptic PDE in
% divergence form with homogeneous Dirichlet boundary conditions:
%
%   div(f_0 grad u)=s, in O
%   u=0, on boundary of O
%
% where s:O -> R is a given C^\infty(O) source function and 
% f_0:O -> [f_min,+infty), f_min>0, is the unknown (sufficiently smooth)
% diffusivity.
%
% There exists a unique classical solution G(f_0)=u_f0. We observe data
%
%   Y_i = u_f0(X_i) + sigma W_i,   i = 1,...,n
%
% where X_i are uniform random locations in O, sigma>0 and W_i are i.i.d. 
% N(0,1) random variables.
%
% The following code generates n observations (Y_i,X_i), i=1,...,n

%%
% Specify true diffusivity f_0

% Display more digits
format long

% Specify radius of disk
r = 1/sqrt(pi);

% Specify f_0 as a function of (x,y)
f_min = 1;
f0 = @(x,y) f_min + exp(-(10*x-2.5).^2-(10*y-2.5).^2)...
    + exp(-(10*x-2.5).^2-(10*y+2.5).^2)...
    + exp(-(10*x+2.5).^2-(10*y+2.5).^2)...
    +exp(-(10*x+2.5).^2-(10*y-2.5).^2);

% Plot f_0 (possibly as a surface)
nRhos = linspace(0,r,500) ;
nThetas = linspace(0,2*pi,500) ;
[Rhos, Thetas] = meshgrid(nRhos,nThetas) ;
xx = Rhos.*cos(Thetas); 
yy = Rhos.*sin(Thetas);
f0_surf = f_min + exp(-(10*xx-2.5).^2-(10*yy-2.5).^2)...
    + exp(-(10*xx-2.5).^2-(10*yy+2.5).^2)...
    + exp(-(10*xx+2.5).^2-(10*yy+2.5).^2)...
    + exp(-(10*xx+2.5).^2-(10*yy-2.5).^2);
figure()
surf(xx, yy, f0_surf)
title('True diffusivity f_0','FontSize',20)
%l = light('Position',[-50 -15 29])
%set(gca,'CameraPosition',[208 -50 7687])
xlabel('x','FontSize', 15)
ylabel('y', 'FontSize', 15)
lighting phong
shading interp
colorbar EastOutside
colormap jet % change colormap. Some options are parula (default), jet, 
% hot, cool (default for pdeplot), winter
view(0,90) % to set view from above. Set view(45,45) for view from side
colorbar

%%
% Elliptic PDE solution corresponding to true diffusivity f_0

% Specify true diffusivity as a function of (location,state) 
% to pass to elliptic PDE solver
f0_fun=@(location,state) f0(location.x,location.y);

% Create PDE model
model = createpde();

% Define circular domain O with centre (0,0) and radius r
O = [1,0,0,r]';

% Create geometry
geom = decsg(O);
geometryFromEdges(model,geom);

% Plot geometry
%figure()
%pdegplot(model,'EdgeLabels','on')

% Create mesh
mesh = generateMesh(model,'Hmax',0.05); % reduce Hmax for finer mesh
mesh=model.Mesh.Nodes;
mesh_size=size(mesh);
mesh_size=mesh_size(2);
%figure()
%pdeplot(model)

% Specify zero Dirichlet boundary conditions on all edges
applyBoundaryCondition(model,'dirichlet','Edge',1:model.Geometry.NumEdges,'u',0);

% Specify the coefficients with c=f_0 and f=1 (the sintax for the source in MATLAB)
specifyCoefficients(model,'m',0,'d',0,'c',f0_fun,'a',0,'f',1);

% Solve PDE and plot solution
results = solvepde(model);
u0 = results.NodalSolution; 
figure()
pdeplot(model,'XYData',u0)
title('PDE solution G(f_0)\equiv u_{f_0}','FontSize', 20)
xlabel('x','FontSize', 15)
ylabel('y', 'FontSize', 15)

%%
% Noisy observations of PDE solution and log-likelihood computation

% Sample design points and noise variales
n=1000; % number of observations
sigma=.001; % noise standard deviation
rand_index=sort(randsample(mesh_size,n)); % random indices in the mesh
rand_mesh=mesh(:,rand_index); % random sample of mesh points drawn 
% uniformly at random
figure()
scatter(rand_mesh(1,:), rand_mesh(2,:),'filled') % plot of sampled locations
title('Random design points X_1,...,X_n','FontSize', 20)
xlabel('x','FontSize', 15)
ylabel('y', 'FontSize', 15)

% Sample observations
observations=u0(rand_index)+(mvnrnd(zeros(n,1),sigma^2*eye(n)))'; % add 
% i.i.d N(0,sigma^2) noise to the observation
u0_corrupted=u0;
u0_corrupted(rand_index)=observations;

% Plot corrupted PDE solution
figure()
pdeplot(model,'XYData',u0_corrupted)
title('Observations Y_i=u_{f_0}(X_i)+\sigma W_i','FontSize', 20);
xlabel('x','FontSize', 15)
ylabel('y', 'FontSize', 15)

% Compute log-likelihood of f_0
loglik0=-sum((observations-u0(rand_index)).^2 )/(2*sigma^2);
disp(['Log-likelihood of f_0 = ', num2str(loglik0)])

