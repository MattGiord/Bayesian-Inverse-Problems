% Copyright (c) 2023 Matteo Giordano
%
% Codes accompanying the article "Bayesian nonparametric inference in 
% PDE models: asymptotic theory and implementation" by Matteo Giordano

%%
% Bayesian nonparametric inference for the diffusivity f:O->[f_min,+infty) 
% with Matérn process priors via the pCN algorithm

% Requires output of GenerateObservations.m (including f_0, observations 
% and geometry), and the file K_mat.m (Matérn kernel)

%%
% Mesh for discretisation of parameter space via piecewise linear functions

model_prior = createpde(); 
geometryFromEdges(model_prior,geom);
mesh_prior = generateMesh(model_prior,'Hmax',0.05);
mesh_prior=model_prior.Mesh.Nodes;
M=size(mesh_prior); 
M=M(2); % discretised parameter space dimension

%%
% pCN initialisation

% pCN initialisation at 0
f_init=@(location,state) f_min+exp(0*location.x);
F_init_num = 0*mesh_prior(1,:); % specify the initialisation point
% as numeric vector

% pCN initialisation at f0
%f_init=@(location,state) f0(location.x,location.y);
%F_init_num = log(exp(-(10*mesh_prior(1,:)-2.5).^2-(10*mesh_prior(2,:)-2.5).^2) ...
%    + exp(-(10*mesh_prior(1,:)-2.5).^2-(10*mesh_prior(2,:)+2.5).^2) ...
%    + exp(-(10*mesh_prior(1,:)+2.5).^2-(10*mesh_prior(2,:)+2.5).^2) ...
%    + exp(-(10*mesh_prior(1,:)+2.5).^2-(10*mesh_prior(2,:)-2.5).^2));

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
% Prior covariance matrix for prior draws

% Compute covariance matrix for Matérn process on mesh for discretisation
prior_regularity = 10; l = .125; % hyper-parameters in Mtern kernel
Cov_matr = zeros(M,M);
for i=1:M
    for j=i:M
        Cov_matr(i,j) = K_mat(mesh_prior(:,i),mesh_prior(:,j),prior_regularity,l);
        Cov_matr(j,i) = Cov_matr(i,j);
    end
end

%%
% Sample and plot a prior draw

prior_draw = mvnrnd(zeros(M,1),Cov_matr,1)';
figure()
tri = delaunay(mesh_prior(1,:),mesh_prior(2,:));
subplot(1,2,1)
trisurf(tri,mesh_prior(1,:),mesh_prior(2,:),prior_draw);
title('F\sim \Pi(\cdot)','FontSize',20)
xlabel('x','FontSize', 15)
ylabel('y', 'FontSize', 15)
%l = light('Position',[-50 -15 29])
%set(gca,'CameraPosition',[208 -50 7687])
lighting phong
shading interp
colorbar EastOutside
colormap jet
view(0,90)
colorbar
subplot(1,2,2)
trisurf(tri,mesh_prior(1,:),mesh_prior(2,:),f_min+exp(prior_draw));
title('f=f_{min}+exp(F)','FontSize',20)
xlabel('x','FontSize', 15)
ylabel('y', 'FontSize', 15)
%l = light('Position',[-50 -15 29])
%set(gca,'CameraPosition',[208 -50 7687])
lighting phong
shading interp
colorbar EastOutside
colormap jet
view(0,90)
colorbar

%%
% pCN algorithm

% Parameters for pCN algorithms
delta=.00125; % smaller delta imply shorter moves for the pCN proposal
MCMC_length=10000; % number of pCN draws

% Trackers
accept_count = zeros(1,MCMC_length); % initialise vector % to keep track 
% of acceptance steps
alphas = ones(1,MCMC_length); % initialise vector to keep  track of 
% acceptance probability
unifs = rand(1,MCMC_length); % i.i.d. Un(0,1) random variables for 
% Metropolis-Hastings updates
F_MCMC = zeros(M,MCMC_length); % initialise marix to store the MCMC chain 
% of functions
F_MCMC(:,1)=F_init_num; % set pCN initialisation point
loglik_MCMC = zeros(1,MCMC_length); % initialise vector to keep track of 
% the loglikelihood of the pCN chain
loglik_MCMC(1) = loglik_init; % set initial loglikelihood for 
% initialisation point
loglik_current = loglik_MCMC(1);
F_current_num = F_MCMC(:,1);

tic

for MCMC_index=2:MCMC_length
    disp(["MCMC step n." num2str(MCMC_index)])
    
    % Construct pCN proposal
    prior_draw = mvnrnd(zeros(M,1),Cov_matr,1)';
    F_proposal_num = sqrt(1-2*delta)*F_current_num + sqrt(2*delta)*prior_draw;
        
    % Define proposal diffusivity function to pass to PDE solver
    F_proposal=@(location,state) griddata(mesh_prior(1,:),mesh_prior(2,:),F_proposal_num,location.x,location.y);
    f_proposal=@(location,state) f_min+exp(F_proposal(location,state));

    % Solve PDE with coefficient equal to the proposal
    specifyCoefficients(model,'m',0,'d',0,'c',f_proposal,'a',0,'f',1);
    results_proposal = solvepde(model);
    u_proposal = results_proposal.NodalSolution; 

    % Compute acceptance probability
    loglik_proposal = -sum( (observations-u_proposal(rand_index)).^2 )/(2*sigma^2);
    alpha=exp(loglik_proposal-loglik_current)
    alphas(MCMC_index)=alpha;

    if unifs(MCMC_index)<alpha % if verified, accept proposal
        F_current_num = F_proposal_num;
        loglik_current = loglik_proposal;
        accept_count(1,MCMC_index)=accept_count(1,MCMC_index-1)+1;
    else
        accept_count(1,MCMC_index)=accept_count(1,MCMC_index-1);
    end
    F_MCMC(:,MCMC_index) = F_current_num;
    loglik_MCMC(MCMC_index) = loglik_current;
end

toc % return run time. On MATLAB 2023a running on MacBook Pro M1 it takes 
% ~2min for 1000 iterates, with the PDE solver using a mesh with 1981 nodes and 
% a discretisation of the parameter space using a mesh of 1981 nodes

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
F_mean_num = mean(F_MCMC(:,burnin+1:MCMC_length),2);
f_mean_num=f_min+exp(F_mean_num);
f0_mesh_prior = f_min + exp(-(10*mesh_prior(1,:)-2.5).^2-(10*mesh_prior(2,:)-2.5).^2) ...
    + exp(-(10*mesh_prior(1,:)-2.5).^2-(10*mesh_prior(2,:)+2.5).^2)...
    + exp(-(10*mesh_prior(1,:)+2.5).^2-(10*mesh_prior(2,:)+2.5).^2)...
    + exp(-(10*mesh_prior(1,:)+2.5).^2-(10*mesh_prior(2,:)-2.5).^2);

% Plot MCMC average and estimation erorr
figure()
ax1 = subplot(1,2,1);
trisurf(tri,mesh_prior(1,:),mesh_prior(2,:),f0_mesh_prior) 
shading interp
colorbar EastOutside
view(0,90) % to set view from above
colorbar
caxis([min(f0_mesh_prior), max(f0_mesh_prior)]);
colormap(ax1,jet)
title('True f_0','FontSize',20)
ax2=subplot(1,2,2);
trisurf(tri,mesh_prior(1,:),mesh_prior(2,:),f_mean_num)
xlabel('x','FontSize', 15)
ylabel('y', 'FontSize', 15)
title('Posterior mean estimate (via pCN)','FontSize',20)
shading interp
colorbar EastOutside
view(0,90)
colorbar
colormap(ax2,jet)
caxis([min(f0_mesh_prior), max(f0_mesh_prior)]);
xlabel('x','FontSize', 15)
ylabel('y', 'FontSize', 15)

figure()
ax3 = subplot(1,1,1);
trisurf(tri,mesh_prior(1,:),mesh_prior(2,:),f0_mesh_prior'-f_mean_num)
xlabel('x','FontSize', 15)
ylabel('y', 'FontSize', 15)
title('Estimation Error','FontSize',20)
%l = light('Position',[-50 -15 29]);
%set(gca,'CameraPosition',[208 -50 7687]);
%lighting phong
shading interp
colorbar EastOutside
view(0,90) % to change view from above. Set view(0,90) for view from above
colorbar
colormap(ax3,hot)

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
