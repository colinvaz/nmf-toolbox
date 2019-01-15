function [W, H, cost] = lnmf(V, num_basis_elems, config)
% lnmf Decompose a non-negative matrix V into WH using Local NMF [1] by
% minimizing the KL divergence. W is a basis matrix and H is the encoding
% matrix that encodes the input V in terms of the basis W.
% 
% Inputs:
%   V: [non-negative matrix]
%       m-by-n non-negative matrix containing data to be decomposed.
%   num_basis_elems: [positive scalar]
%       [positive scalar]: number of basis elements (columns of W/rows of H)
%       for 1 source.
%   config: [structure] (optional)
%       structure containing configuration parameters.
%       config.W_init: [non-negative matrix] (default: random matrix)
%           initialize 1 basis matrix for 1 source with a
%           m-by-num_basis_elems matrix.
%       config.H_init: [non-negative matrix] (default: random matrix)
%           initialize 1 encoding matrix for 1 source with a
%           num_basis_elems-by-n non-negative matrix.
%       config.W_fixed: [boolean] (default: false)
%           indicate if the basis matrix is fixed during the update
%           equations.
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
%   W: [non-negative matrix]
%       m-by-num_basis_elems non-negative basis matrix.
%   H: [non-negative matrix]
%       num_basis_elems-by-n non-negative encoding matrix.
%   cost: [vector]
%       value of the cost function after each iteration.
%
% References:
%   [1] S. Z. Li, X. Hou, H. Zhang, and Q. Cheng, "Learning Spatially
%       Localized Parts-Based Representation," in Proc. IEEE Computer Soc.
%       Conf. Computer Vision Pattern Recognition, Hawaii, 2001, pp.
%       207-212.
%
% TODO: Add support for multiple sources
%       Add support for minimizing AB divergence
%
% NMF Toolbox
% Colin Vaz - cvaz@usc.edu
% Signal Analysis and Interpretation Lab (SAIL) - http://sail.usc.edu
% University of Southern California
% 2015

% Check if configuration structure is given
if nargin < 3
	config = struct;
end

[m, n] = size(V);
config = ValidateParameters(V, num_basis_elems, config);

W = config.W_init;
W = W * diag(1 ./ sum(W, 1));
H = config.H_init;

V_hat = ReconstructFromDecomposition(W, H);

cost = zeros(config.maxiter, 1);

for iter = 1 : config.maxiter
	% Update basis matrix
    if ~config.W_fixed
    	W = W .* (((V ./ V_hat) * H') ./ max(ones(m, n) * H', eps));
        W = W * diag(1 ./ sum(W, 1));
        V_hat = ReconstructFromDecomposition(W, H);
    end	

	% Update encoding matrix
    if ~config.H_fixed
    	H = sqrt(H .* (W' * (V ./ V_hat)));
        V_hat = ReconstructFromDecomposition(W, H);
    end
    
    % Calculate cost for this iteration
    cost(iter) = sum(sum(V .* log(V ./ V_hat) - V + V_hat));
    
    % Stop iterations if change in cost function less than the tolerance
    if iter > 1 && cost(iter) <= cost(iter-1) && cost(iter-1) - cost(iter) <= config.tolerance
        break;
    end
end

end  % function lnmf

function config_out = ValidateParameters(V, num_basis_elems, config_in)
% ValdidateParameters private function
% Check parameters supplied by the user and fill in default values for
% unsupplied parameters.

config_out = config_in;

[data_dim, num_samples] = size(V);

% Initialize encoding matrices
if ~isfield(config_out, 'H_init') || isempty(config_out.H_init)  % not given any inital encoding matrices. Fill these in.
    config_out.H_init = max(rand(num_basis_elems, num_samples), eps);
end

% Initialize basis matrices
if ~isfield(config_out, 'W_init') || isempty(config_out.W_init)  % not given any inital basis matrices. Fill these in.
    config_out.W_init = max(rand(data_dim, num_basis_elems), eps);
    config_out.W_init = config_out.W_init * diag(1 ./ sum(config_out.W_init, 1));
end

% Update switches for basis matrices
if ~isfield(config_out, 'W_fixed') || isempty(config_out.W_fixed)  % not given an update switch. Fill this in.
    config_out.W_fixed = false;
end

% Update switches for encoding matrices
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
