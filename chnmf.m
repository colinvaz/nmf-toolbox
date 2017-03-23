function [W, H, S, G, cost] = chnmf(V, num_basis_elems, config)
% chnmf Decompose a matrix V into SGH using Convex hull-NMF (CH-NMF) [1] by
% minimizing the Euclidean distance between V and SGH. W = SG is a basis
% matrix, where the columns of G form convex combinations of S, which
% contain the convex hull of the data V, and H is the encoding matrix that
% encodes the input V in terms of the basis W. Unlike NMF, V can have mixed
% sign. The columns of W can be interpreted as cluster centroids (there is
% a connection to K-means clustering), while H shows the soft membership of
% each data point to the clusters.
%
% Inputs:
%   V: [matrix]
%       m-by-n matrix containing data to be decomposed.
%   num_basis_elems: [positive scalar]
%       number of basis elements (columns of G/rows of H) for 1 source.
%   config: [structure] (optional)
%       structure containing configuration parameters.
%       config.S_init: [matrix] (default: matrix returned by Matlab's
%           convhull function with input V)
%           initialize m-by-p matrix containing p points belonging to the
%           convex hull of V. 
%       config.G_init: [non-negative matrix] (default: random matrix)
%           initialize convex combination matrix with a p-by-num_basis_elems
%           matrix.
%       config.H_init: [non-negative matrix] (default: n indicator vectors
%           of cluster membership using K-means + 0.2)
%           initialize encoding matrix with a num_basis_elems-by-n
%           non-negative matrix.
%       config.G_fixed: [boolean] (default: false)
%           indicate if the convex combination matrix is fixed during the
%           update equations.
%       config.H_fixed: [boolean] (default: false)
%           indicate if the encoding matrix is fixed during the update
%           equations.
%       config.G_sparsity: [non-negative scalar] (default: 0)
%           sparsity level for the convex combination matrix.
%       config.pct_eigval_energy: [scalar in (0, 1]] (default: 0.95)
%           use eigenvectors corresponding to eigenvalues that account for
%           a given percentage of variance in the data for doing
%           projections to find the convex hull of the data.
%       config.maxiter: [positive scalar] (default: 100)
%           maximum number of update iterations.
%       config.tolerance: [positive scalar] (default: 1e-3)
%           maximum change in the cost function between iterations before
%           the algorithm is considered to have converged.
%
% Outputs:
%   W: [matrix]
%       m-by-num_basis_elems basis matrix. W = S*G.
%   H: [non-negative matrix]
%       num_basis_elems-by-n non-negative encoding matrix.
%   S: [matrix]
%       m-by-p matrix of p points belonging to the convex hull of V.
%   G: [non-negative matrix]
%       p-by-num_basis_elems matrix of convex combinations of the columns
%       of S.
%   cost: [vector]
%       value of the cost function after each iteration.
%
% References:
%   [1] C. Thurau, K. Kersting, M. Wahabzada, and C. Bauckhage, "Convex
%       non-negative matrix factorization for massive datasets," Knowledge
%       and Information Systems, vol. 29, no. 2, pp. 457-478, Nov. 2011.
%
% NMF Toolbox
% Colin Vaz - cvaz@usc.edu
% Signal Analysis and Interpretation Lab (SAIL) - http://sail.usc.edu
% University of Southern California
% 2015

% Check if configuration structure is given.
if nargin < 3
	config = struct;
end

[m, n] = size(V);

% Set percentage of eigenvalue energy to keep when doing projections for
% calculating convex hull of data
if ~isfield(config, 'pct_eigval_energy') || config.pct_eigval_energy < 0 || config.pct_eigval_energy > 1
    config.pct_eigval_energy = 0.95;
end

% Initialize convex hull points
if ~isfield(config, 'S_init') || isempty(config.S_init)
    % If V is 1D, then convexhull is just max and min points
    if m == 1
        config.S_init = [min(V) max(V)];
    else
        data_cov = cov(V');
        [eigenvecs, eigenvals] = eig(data_cov);
        eigenvals = diag(eigenvals(end:-1:1, end:-1:1));  % order eigenvalues from largest to smallest
        eigenvecs = eigenvecs(:, end:-1:1);  % reorder corresponding eigenvectors
        num_eigvals_keep = min(find(cumsum(eigenvals.^2) / sum(eigenvals' * eigenvals) > config.pct_eigval_energy));
        num_eigvals_keep = max(num_eigvals_keep, 2);  % keep at least the first 2 eigenvalues
        config.S_init = [];
        for e1 = 1 : num_eigvals_keep-1
            for e2 = e1+1 : num_eigvals_keep
                projected_data = V' * [eigenvecs(:, e1) eigenvecs(:, e2)];
                convexhull_idx = convhull(projected_data);
                config.S_init = [config.S_init V(:, convexhull_idx)];
                config.S_init = unique(config.S_init.', 'rows').';  % remove duplicate data points
            end
        end
    end
end
S = config.S_init;
p = size(S, 2);

% Initialize convex combination matrix
if ~isfield(config, 'G_init') || isempty(config.G_init)
    config.G_init = rand(p, num_basis_elems);    
end
G = config.G_init;
G = G * diag(1 ./ sum(G, 1));

% Update switch for convex combination matrix
if ~isfield(config, 'G_fixed')
    config.G_fixed = false;
end

% Sparsity level for convex combination matrix
if ~isfield(config, 'G_sparsity') || isempty(config.G_sparsity)
    config.G_sparsity = 0;
end

% Initialize encoding matrix
if ~isfield(config, 'H_init') || isempty(config.H_init)
%     cluster_idx = kmeans(V.', num_basis_elems);
%     config.H_init = zeros(num_basis_elems, n);
%     for j = 1 : n
%         config.H_init(cluster_idx(j), j) = 1;
%     end
%     config.H_init = config.H_init + 0.2*rand(num_basis_elems, n);
    config.H_init = rand(num_basis_elems, n);
end
H = config.H_init;

% Update switch for encoding matrix
if ~isfield(config, 'H_fixed')
    config.H_fixed = false;
end

% Sparsity level for encoding matrix
% TODO: look into using Hoyer's sparsity constraint
if ~isfield(config, 'H_sparsity') || isempty(config.H_sparsity)
    config.H_sparsity = 0;
% elseif config.H_sparsity > 0  % Hoyer's sparsity constraint
% %     L1s = sqrt(n) - (sqrt(n) - 1) * config.H_sparsity;
%     L1s = sqrt(num_basis_elems) - (sqrt(num_basis_elems) - 1) * config.H_sparsity;
%     for k = 1 : n%um_basis_elems
%         H_norm = norm(H(:, k));
%         H(:, k) = H(:, k) / H_norm;
%         H(:, k) = (projfunc(H(:, k), L1s, 1, 1))';
%         H(:, k) = H(:, k) * H_norm;
%     end
end

% Maximum number of update iterations
if ~isfield(config, 'maxiter') || config.maxiter <= 0
    config.maxiter = 100;
end

% Maximum tolerance in cost function change per iteration
if ~isfield(config, 'tolerance') || config.tolerance <= 0
    config.tolerance = 1e-3;
end

S_V_pos = 0.5 * (abs(S' * V) + (S' * V));
S_V_neg = 0.5 * (abs(S' * V) - (S' * V));
S_S_pos = 0.5 * (abs(S' * S) + (S' * S));
S_S_neg = 0.5 * (abs(S' * S) - (S' * S));
W = S * G;

cost = zeros(config.maxiter, 1);

for iter = 1 : config.maxiter
    % Update convex combination matrix
    if ~config.G_fixed
        G = G .* (((S_V_pos + S_S_neg * G * H) * H') ./ max((S_V_neg + S_S_pos * G * H) * H' + config.G_sparsity, eps));
        G = G * diag(1 ./ sum(G, 1));
    end
    W = S * G;

    % Update encoding matrix
    if ~config.H_fixed
        H = H .* ((S_V_pos + S_S_neg * G * H) ./ max(S_V_neg + S_S_pos * G * H + config.H_sparsity, eps));
    end
    
    % Calculate cost for this iteration
    V_hat = ReconstructFromDecomposition(W, H);
    cost(iter) = 0.5 * sum(sum((V - V_hat).^2));
    
    % Stop iterations if change in cost function less than the tolerance
    if iter > 1 && cost(iter) < cost(iter-1) && cost(iter-1) - cost(iter) < config.tolerance
        cost = cost(1 : iter);  % trim vector
        break;
    end
end
