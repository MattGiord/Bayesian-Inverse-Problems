% Copyright (c) 2023 Matteo Giordano
%
% Codes accompanying the article "Bayesian nonparametric inference in 
% PDE models: asymptotic theory and implementation" by Matteo Giordano

%%
% Bayesian nonparametric inference for the diffusivity f:O->[f_min,+infty) 
% with truncated Gaussian series priors on the Dirichlet Laplacian eigebasis
% via the pCN algorithm

% Requires output of GenerateObservations.m (including f_0, observations 
% and geometry)

%%
% Mesh for computation of the Dirichlet-Laplacian eigenpairs

model_prior = createpde(); 
geometryFromEdges(model_prior,geom);
mesh_prior = generateMesh(model_prior,'Hmax',0.075);
mesh_prior=model_prior.Mesh.Nodes;
mesh_prior_size=size(mesh_prior); 
mesh_prior_size=mesh_prior_size(2); % discretised parameter space dimension

%%
% Solve elliptic eigenvalue problem for the Dirichlet-Laplacian

tic

% Specity homogeneous Dirichlet boundary conditions
applyBoundaryCondition(model_prior,'dirichlet','Edge',1:model.Geometry.NumEdges,'u',0); 
% Specify coefficients for eigenvalue equation
specifyCoefficients(model_prior,'m',0,'d',1,'c',1,'a',0,'f',1);
range = [-1,1000]; % range to search for eigenvalues
results = solvepdeeig(model_prior,range); % solve eigenvalue equation
lambdas_basis = results.Eigenvalues; % extract eigenvalues
J_basis = length(lambdas_basis); % number of eigenvalues (dimension of
% discretised parameter space)
e_basis = results.Eigenvectors; % extract eigenfunctions

toc

figure() 
subplot(1,3,1)
pdeplot(model_prior,'XYData',e_basis(:,1)); % plot first eigenfunction
title('e_0','FontSize',15)
subplot(1,3,2)
pdeplot(model_prior,'XYData',e_basis(:,2)); % plot second eigenfunction
title('e_2','FontSize',15)
subplot(1,3,3)
pdeplot(model_prior,'XYData',e_basis(:,J_basis)); % plot eigenfunction 
% corresponding to the largest found eigenvalue
title('e_J','FontSize',15)

% Plot the eigenvalues

figure()
axes('FontSize', 15, 'NextPlot','add')
plot(lambdas_basis,'.','Linewidth',3)
xlabel('j', 'FontSize', 15);
ylabel('\lambda_j', 'FontSize', 15);

%%
%L2 normalisation of eigenfunction

% The eigenfunction evaluations over the mesh returned by the solver are 
% normalised in a way that the sum of the squared values are equal to 1. Hence, 
% the value of the eigenfunctions returned is connect to the mesh size: 
% the more points in the mesh, the smaller the values
%for i=1:J_basis
%    sum(e_basis(:,i).^2) % = 1 for each eigenfunction
%    mean(e_basis(:,i).^2) % = 1/mesh_size for each eigenfunction
%end

% L^2-normalisation of eigenfunctions
%for j=1:J
    % Put e_basis(:,j).^2 in polar coordinates. Need to add 1 to 
    % avoid numerical instability
    %ejpolar2 = @(thet,rad) (1+griddata(mesh_nodes(1,:),mesh_nodes(2,:),e_basis(:,j),...
    %rad.*cos(thet),rad.*sin(thet)).^2).*rad; 
    % Compute squared L2 norm of e_basis(:,j). Need to remove pi*(r*.999)^2 
    % to account for addition of 1 above
    %sqnorm = integral2(ejpolar2,0,2*pi,0,r*.9995)- pi*(r*.9995)^2 
    %e_basis(:,j) = e_basis(:,j)/sqrt(sqnorm); % normalise e_basis(:,j) to 
    % have unit L2 norm
%end

vol = pi*r^2; % area of circular domain

% For a faster normalisation, normalise the eigenfunctions so that
% the mean(e_basis(:,j).^2) = 1/sqrt(vol), thereby approximating the unit
% L^2-norm normalisation over the disk. Since originally 
% mean(e_basis(:,i).^2) = 1/mesh_prior_size, we need to multiply 
% e_basis(:,j) by sqrt(mesh_prior_size/area)
e_basis(:,1:J_basis) = e_basis(:,1:J_basis)*sqrt(mesh_prior_size/vol);

%for i=1:J_basis
%    mean(e_basis(:,i).^2) % = 1/sqrt(vol)
%end

figure() 
subplot(1,3,1)
pdeplot(model_prior,'XYData',e_basis(:,1));
title('Normalised e_1','FontSize',15)
subplot(1,3,2)
pdeplot(model_prior,'XYData',e_basis(:,2)); 
title('Normalised e_2','FontSize',15)
subplot(1,3,3)
pdeplot(model_prior,'XYData',e_basis(:,J_basis)); 
title('Normalised e_D','FontSize',15)

%%
% Projection of F_0 onto the Dirichlet-Laplacian eigenbasis

mesh_prior_elements = model_prior.Mesh.Elements;
mesh_prior_elements_num = size(mesh_prior_elements); 
mesh_prior_elements_num = mesh_prior_elements_num(2); % number of triangular
% mesh elements
[~,elements_area] = area(model_prior.Mesh); % area of triangular mesh elements

% Compute barycenters of triangular mesh elements
barycenters = zeros(2,mesh_prior_elements_num);
for i=1:mesh_prior_elements_num
    barycenters(:,i) = mean(mesh_prior(:,mesh_prior_elements(1:3,i)),2);
end

f0_mesh_prior = f0(mesh_prior(1,:),mesh_prior(2,:));
F0_mesh_prior = log(f0_mesh_prior);
F0_interp=scatteredInterpolant(mesh_prior(1,:)',mesh_prior(2,:)',F0_mesh_prior');
F0_bary=F0_interp(barycenters(1,:),barycenters(2,:));

F0_coeff=zeros(J_basis,1); % initialises vector to store the Fourier 
% coefficients of F0 in the Dirichlet-Laplacian eigenbasis

for j=1:J_basis
    ej_basis_interp=scatteredInterpolant(mesh_prior(1,:)',mesh_prior(2,:)',e_basis(:,j));
    ej_basis_bary=ej_basis_interp(barycenters(1,:),barycenters(2,:));
    F0_coeff(j)=sum(elements_area.*F0_bary.*ej_basis_bary);
end

F0_proj=zeros(1,mesh_prior_size);
for j=1:J_basis
    F0_proj = F0_proj+F0_coeff(j)*e_basis(:,j)';
end

figure()
subplot(1,3,1)
pdeplot(model_prior,'XYData',F0_mesh_prior,'ColorMap',jet)
title('True F_0','FontSize',20)
colorbar('Fontsize',15)

subplot(1,3,2)
pdeplot(model_prior,'XYData',F0_proj,'ColorMap',jet)
title('Projection of F_0','FontSize',20)
colorbar('Fontsize',15)

subplot(1,3,3)
pdeplot(model_prior,'XYData',F0_mesh_prior-F0_proj,'ColorMap',jet)
title('Approximation error','FontSize',20)
colorbar('Fontsize',15)


%%
% pCN initialisation

% pCN initialisation at 0
theta_init = zeros(J_basis,1);
F_init_num=zeros(1,mesh_prior_size);
f_init_num=exp(F_init_num);

% pCN initialisation at f_0
%theta_init = F0_coeff;
%F_init=F0_proj;
%f_init=f_min+ exp(F_init);

% Specify f_init as a function of (location,state) to pass to elliptic PDE solver
F_init=@(location,state) griddata(mesh_prior(1,:),mesh_prior(2,:),F_init_num,location.x,location.y);
f_init=@(location,state) f_min+exp(F_init(location,state));

% Solve elliptic PDE for initial pCN state
specifyCoefficients(model,'m',0,'d',0,'c',f_init,'a',0,'f',1);
results_current = solvepde(model);
u_current = results_current.NodalSolution; 

% Compute likelihood for initialisation point and compare it to likelihood
% of f_0
loglik_init = -sum( (observations-u_current(rand_index)).^2 )/(2*sigma^2);
disp(['Log-likelihood of pCN initialisation = ', num2str(loglik_init)])
disp(['Log-likelihood of f_0 = ', num2str(loglik0)])

%%
% Prior covariance matrix for Gaussian series prior draws

prior_regularity=.625; 
prior_cov=diag(lambdas_basis.^(-2*prior_regularity)); % diagonal prior covariance matrix

%%
% Sample and plot a prior draw

theta_rand=mvnrnd(zeros(J_basis,1),prior_cov,1)'; % sample Fourier 
% coefficients from prior

F_rand=zeros(1,mesh_prior_size);

for j=1:J_basis
    F_rand = F_rand+theta_rand(j)*e_basis(:,j)';
end

f_rand=exp(F_rand);

figure()
subplot(1,2,1)
pdeplot(model_prior,'XYData',F_rand,'ColorMap',jet);
title('F\sim\Pi(\cdot)','FontSize',15)
subplot(1,2,2)
pdeplot(model_prior,'XYData',f_rand,'ColorMap',jet);
title('f=e^F','FontSize',15)

%%
% pCN algorithm

% Parameters for pCN algorithms
delta=.0001; % smaller delta imply shorter moves for the pCN proposal
MCMC_length=10000; % number of pCN draws

% Trackers
accept_count = zeros(1,MCMC_length); % initialise vector % to keep track 
% of acceptance steps
alphas = ones(1,MCMC_length); % initialise vector to keep  track of 
% acceptance probability
unifs = rand(1,MCMC_length); % i.i.d. Un(0,1) random variables for 
% Metropolis-Hastings updates
theta_MCMC = zeros(J_basis,MCMC_length); % initialise marix to store the 
% MCMC chain of Fourier coefficients
theta_MCMC(:,1)=theta_init; % set pCN initialisation point
loglik_MCMC = zeros(1,MCMC_length); % initialise vector to keep track of 
% the loglikelihood of the pCN chain
loglik_MCMC(1) = loglik_init; % set initial loglikelihood for 
% initialisation point
loglik_current = loglik_MCMC(1);
theta_current = theta_MCMC(:,1);

tic

for MCMC_index=2:MCMC_length
    disp(["MCMC step n." num2str(MCMC_index)])
    
    % Construct pCN proposal
    theta_rand=mvnrnd(zeros(J_basis,1),prior_cov,1)';
    theta_proposal= sqrt(1-2*delta)*theta_current + sqrt(2*delta)*theta_rand;

    F_proposal_num=zeros(mesh_prior_size,1);
    for j=1:J_basis
        F_proposal_num = F_proposal_num+theta_proposal(j)*e_basis(:,j);
    end
        
    % Define proposal diffusivity function to pass to PDE solver
    F_proposal=@(location,state) griddata(mesh_prior(1,:),mesh_prior(2,:),F_proposal_num,location.x,location.y);
    f_proposal=@(location,state) exp(F_proposal(location,state));

    % Solve PDE with coefficient equal to the proposal
    specifyCoefficients(model,'m',0,'d',0,'c',f_proposal,'a',0,'f',1);
    results_proposal = solvepde(model);
    u_proposal = results_proposal.NodalSolution; 

    % Compute acceptance probability
    loglik_proposal = -sum( (observations-u_proposal(rand_index)).^2 )/(2*sigma^2);
    alpha=exp(loglik_proposal-loglik_current)
    alphas(MCMC_index)=alpha;

    if unifs(MCMC_index)<alpha % if verified, accept proposal
        theta_current = theta_proposal;
        loglik_current = loglik_proposal;
        accept_count(1,MCMC_index)=accept_count(1,MCMC_index-1)+1;
    else
        accept_count(1,MCMC_index)=accept_count(1,MCMC_index-1);
    end
    theta_MCMC(:,MCMC_index) = theta_current;
    loglik_MCMC(MCMC_index) = loglik_current;
end

toc % return run time. On MATLAB 2023a running on MacBook Pro M1 it takes 
% ~1min for 1000 iterates, with the PDE solver using a mesh with 1981 nodes and 
% a discretisation of the parameter space using 69 basis functions

%%
% Acceptance and loglikelihood along the MCMC chain

n_accept_steps=accept_count(1,MCMC_length);
accept_ratio=accept_count./(1:MCMC_length);

figure()
subplot(1,2,1)
plot(accept_ratio,'LineWidth',2)
title('Acceptance ratio','FontSize',20)
xlabel('MCMC step','FontSize', 15)
subplot(1,2,2)
plot(loglik_MCMC,'LineWidth',2)
%title('Loglikelihood along the MCMC chain','FontSize',20)
xlabel('MCMC step','FontSize', 15)
yline(loglik0,'r','LineWidth',2)
legend('Loglikelihood along MCMC chain','Loglikelihood of f_0','Fontsize',15)

%%
% MCMC average and estimation error

% Compute MCMC average
burnin=1000;
theta_mean = mean(theta_MCMC(:,burnin+1:MCMC_length),2);
F_mean_num=zeros(mesh_prior_size,1);
    for j=1:J_basis
        F_mean_num = F_mean_num+theta_mean(j)*e_basis(:,j);
    end
f_mean_num=exp(F_mean_num);

% Plot MCMC average and estimation erorr

figure()
subplot(1,2,1)
pdeplot(model_prior,'XYData',f0_mesh_prior,'ColorMap',jet)
title('True f_0','FontSize',20)
caxis([min(f0_mesh_prior),max(f0_mesh_prior)])
subplot(1,2,2)
pdeplot(model_prior,'XYData',f_mean_num,'ColorMap',jet)
title('Posterior mean estimate (via pCN)','FontSize',20)
caxis([min(f0_mesh_prior),max(f0_mesh_prior)])

figure()
pdeplot(model_prior,'XYData',f0_mesh_prior-f_mean_num','ColorMap',jet)
title('Estimation error','FontSize',20)

%%
% Compute L^2-estimation error

mesh_prior_elements = model_prior.Mesh.Elements;
mesh_prior_elements_num = size(mesh_prior_elements); 
mesh_prior_elements_num = mesh_prior_elements_num(2); % number of triangular
% mesh elements
[~,elements_area] = area(model_prior.Mesh); % area of triangular mesh elements

% Compute barycenters of triangular mesh elements
barycenters = zeros(2,mesh_prior_elements_num);
for i=1:mesh_prior_elements_num
    barycenters(:,i) = mean(mesh_prior(:,mesh_prior_elements(1:3,i)),2);
end

% Compute piecewise constant approximations of f_0 and posterior mean at
% the triangle baricenters
f0_interp = griddata(mesh_prior(1,:),mesh_prior(2,:),f0_mesh_prior,barycenters(1,:),barycenters(2,:));
f_mean_interp = griddata(mesh_prior(1,:),mesh_prior(2,:),f_mean_num,barycenters(1,:),barycenters(2,:));

% Approximate L^2 distance between f_0 and posterior mean
estim_error = sqrt(sum((f0_interp-f_mean_interp).^2.*elements_area))
