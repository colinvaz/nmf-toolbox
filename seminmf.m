function [W, H, cost] = seminmf(V, num_basis_elems, config)
% seminmf Decompose a matrix V into WH using Semi-NMF [1] by minimizing
% the Euclidean distance between V and WH. W is a basis matrix and H is the
% encoding matrix that encodes the input V in terms of the basis W. Unlike
% NMF, V can have mixed sign. The columns of W can be interpreted as
% cluster centroids (there is a connection between Semi-NMF and K-means
% clustering), while H shows the soft membership of each data point to the
% clusters.
%
% Inputs:
%   V: [matrix]
%       m-by-n matrix containing data to be decomposed.
%   num_basis_elems: [positive scalar]
%       number of basis elements (columns of W/rows of H) for 1 source.
%   config: [structure] (optional)
%       structure containing configuration parameters.
%       config.W_init: [matrix] (default: random matrix)
%           initialize 1 basis matrix with a m-by-num_basis_elems matrix.
%       config.H_init: [non-negative matrix] (default: n indicator vectors
%           of cluster membership using K-means + 0.2)
%           initialize 1 encoding matrix with a num_basis_elems-by-n
%           non-negative matrix.
%       config.W_fixed: [boolean] (default: false)
%           indicate if the basis matrix is fixed during the update equations.
%       config.H_fixed: [boolean] (default: false)
%           indicate if the encoding matrix is fixed during the update
%           equations.
%       config.maxiter: [positive scalar] (default: 100)
%           maximum number of update iterations.
%       config.tolerance: [positive scalar] (default: 1e-3)
%           maximum change in the cost function between iterations before
%           the algorithm is considered to have converged.
%
% Outputs:
%   W: [matrix]
%       m-by-num_basis_elems basis matrix.
%   H: [non-negative matrix]
%       num_basis_elems-by-n non-negative encoding matrix.
%   cost: [vector]
%       value of the cost function after each iteration.
%
% References:
%   [1] C. Ding, T. Li, and M. I. Jordan, "Convex and Semi-Nonnegative
%       Matrix Factorizations," IEEE Trans. Pattern Analysis Machine
%       Intelligence, vol. 32, no. 1, pp. 45-55, Jan. 2010.
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

config = ValidateParameters(V, num_basis_elems, config);

W = config.W_init;
H = config.H_init;

cost = zeros(config.maxiter, 1);

for iter = 1 : config.maxiter
    % Update basis matrix
    if ~config.W_fixed
        W = V * H' / (H * H');
    end
    
    % Update encoding matrix
    if ~config.H_fixed
        W_V_pos = 0.5 * (abs(W' * V) + W' * V);
        W_V_neg = 0.5 * (abs(W' * V) - W' * V);
        W_W_pos = 0.5 * (abs(W' * W) + W' * W);
        W_W_neg = 0.5 * (abs(W' * W) - W' * W);
        H = H .* sqrt((W_V_pos + W_W_neg * H) ./ (W_V_neg + W_W_pos * H));
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

% Prepare the output
% w_length = sqrt(sum(W.^2, 1));
% W = W * diag(1 ./ w_length);  % normalize columns to unit L2 norm
% H = diag(w_length) * H;

end  % function seminmf


function config_out = ValidateParameters(V, num_basis_elems, config_in)
% ValdidateParameters private function
% Check parameters supplied by the user and fill in default values for
% unsupplied parameters.

config_out = config_in;

[data_dim, num_samples] = size(V);

% Initialize encoding matrix
if ~isfield(config_out, 'H_init') || isempty(config_out.H_init)  % not given any inital encoding matrices. Fill these in.
    % TODO: check if this works for multiple dictionaries/activation matrices
    cluster_idx = kmeans(V.', num_basis_elems);
    config_out.H_init = zeros(num_basis_elems, num_samples);
    for sample_count = 1 : num_samples
        config_out.H_init(cluster_idx(sample_count), sample_count) = 1;
    end
    config_out.H_init = config_out.H_init + 0.2;
end

% Initialize basis matrix
if ~isfield(config_out, 'W_init') || isempty(config_out.W_init)  % not given any inital basis matrices. Fill these in.
    config_out.W_init = 2*rand(data_dim, num_basis_elems) - 1;
end

% Update switch for basis matrix
if ~isfield(config_out, 'W_fixed') || isempty(config_out.W_fixed)  % not given an update switch. Fill this in.
    config_out.W_fixed = false;
end

% Update switch for encoding matrix
if ~isfield(config_out, 'H_fixed') || isempty(config_out.H_fixed)  % not given an update switch. Fill this in.
    config_out.H_fixed = false;
end

% Maximum number of update iterations
if ~isfield(config_out, 'maxiter') || config_out.maxiter <= 0
    config_out.maxiter = 100;
end

% Maximum tolerance in cost function change per iteration
if ~isfield(config_out, 'tolerance') || config_out.tolerance <= 0
    config_out.tolerance = 1e-3;
end

end  % function ValidateParameters
