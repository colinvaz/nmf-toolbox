function [W, H, P, cost] = cmfwisa(V, num_basis_elems, config)
% cmfwisa Decompose a (complex-valued) matrix V into WH.*P using Complex
% NMF with intra-source additivity (CMF-WISA) [1] by minimizing the
% Euclidean distance between V and WH.*P. W is a basis matrix, H is the
% encoding matrix that encodes the input V in terms of the basis W, and P
% is the phase matrix. This function can output multiple
% basis/encoding/phase matrices for multiple sources, each of which can 
% be fixed to a given matrix or have a given sparsity level. With 1 source,
% CMF-WISA essentially becomes NMF.
%
% Inputs:
%   V: [matrix]
%       m-by-n matrix containing data, possibly complex-valued, to be decomposed.
%   num_basis_elems: [positive scalar] or [cell array]
%       [positive scalar]: number of basis elements (columns of W/rows of H)
%       for 1 source.
%       [cell array]: K-length array of positive scalars {num_basis_elems_1,
%       ...,num_basis_elems_K} specifying the number of basis elements for
%       K sources.
%   config: [structure] (optional)
%       structure containing configuration parameters.
%       config.W_init: [non-negative matrix] or [cell array] (default:
%           random matrix or K-length cell array of random matrices)
%           [non-negative matrix]: initialize 1 basis matrix for 1 source
%           with a m-by-num_basis_elems tensor.
%           [cell array]: initialize K basis matrices for K sources with a 
%           K-length array containing {m-by-num_basis_elems_1, ...,
%           m-by-num_basis_elems_K} non-negative matrices. 
%       config.H_init: [non-negative matrix] or [cell array] (default:
%           random matrix or K-length cell array of random matrices)
%           [non-negative matrix]: initialize 1 encoding matrix for 1
%           source with a num_basis_elems-by-n non-negative matrix.
%           [cell array]: initialize K encoding matrices for K sources with
%           a K-length array containing {num_basis_elems_1-by-n, ...,
%           num_basis_elems_K-by-n} non-negative matrices.
%       config.P_init: [matrix] or [cell array] (default: exp(1j * arg(V))
%           or K-length cell array of exp(1j * arg(V)) replicated K times)
%           [matrix]: initialize 1 phase matrix for 1 source with a m-by-n
%           (complex-valued) matrix.
%           [cell array]: initialize K phase matrices for K sources with a
%           K-length array containing {m-by-n,...,m-by-n} (complex-valued)
%           matrices.
%       config.W_sparsity: [non-negative scalar] or [cell array] (default: 0)
%           [non-negative scalar]: sparsity level for all basis matrices.
%           [cell array]: K-length array of non-negative scalars indicating
%           the sparsity level of the K basis matrices.
%       config.H_sparsity: [non-negative scalar] or [cell array] (default: 0)
%           [non-negative scalar]: sparsity level for all K encoding matrices.
%           [cell array]: K-length array of non-negative scalars indicating
%           the sparsity level of the K encoding matrices.
%       config.W_fixed: [boolean] or [cell array] (default: false)
%           [boolean]: indicate if all basis matrices are fixed during the
%           update equations.
%           [cell array] K-length array of booleans indicating if the
%           corresponding basis matrices are fixed during the update equations.
%       config.H_fixed: [boolean] or [cell array] (default: false)
%           [boolean]: indicate if all encoding matrices are fixed during
%           the update equations.
%           [cell array] K-length array of booleans indicating if the
%           corresponding encoding matrices are fixed during the update equations.
%       config.P_fixed: [boolean] or [cell array] (default: false)
%           [boolean]: indicate if all phase matrices are fixed during
%           the update equations.
%           [cell array] K-length array of booleans indicating if the
%           corresponding phase matrices are fixed during the update equations.
%       config.maxiter: [positive scalar] (default: 100)
%           maximum number of update iterations.
%       config.tolerance: [positive scalar] (default: 1e-3)
%           maximum change in the cost function between iterations before
%           the algorithm is considered to have converged.
%
% Outputs:
%   W: [non-negative matrix] or [cell array]
%       [non-negative matrix]: m-by-num_basis_elems non-negative basis matrix.    
%       [cell array]: K-length array containing {m-by-num_basis_elems_1, ...,
%       m-by-num_basis_elems_K} non-negative basis matrices for K sources.
%   H: [non-negative matrix] or [cell array]
%       [non-negative matrix]: num_basis_elems-by-n non-negative encoding matrix.
%       [cell array]: K-length array containing {num_basis_elems_1-by-n, ...,
%       num_basis_elems_K-by-n} non-negative encoding matrices.
%   P: [matrix] or [cell array]
%       [matrix]: m-by-n (complex-valued) phase matrix.
%       [cell array]: K-length array containing {m-by-n,...,m-by-n}
%       (complex-valued) phase matrices.
%   cost: [vector]
%       value of the cost function after each iteration.
%
% References:
%   [1] B. King, "New Methods of Complex Matrix Factorization for
%       Single-Channel Source Separation and Analysis," Ph.D. thesis,
%       University of Washington, Seattle, WA, 2012.
%
% NMF Toolbox
% Colin Vaz - cvaz@usc.edu
% Signal Analysis and Interpretation Lab (SAIL) - http://sail.usc.edu
% University of Southern California
% 2015

if nargin < 3
    config = struct;
end

[m, n] = size(V);
if ~iscell(num_basis_elems)
    num_basis_elems = {num_basis_elems};
end
num_sources = length(num_basis_elems);
[config, is_W_cell, is_H_cell] = ValidateParameters('cmfwisa', config, V, num_basis_elems);

% Initialize phase matrices
if ~isfield(config, 'P_init') || isempty(config.P_init)  % not given any inital phase matrices. Fill these in.
    if num_sources == 1
        is_P_cell = false;
    else
        is_P_cell = true;
    end
    config.P_init = cell(num_sources, 1);
    for i = 1 : num_sources
        config.P_init{i} = exp(1j * angle(V)); %V ./ abs(V);
    end
elseif iscell(config.P_init) && length(config.P_init) ~= num_sources  % given an incorrect number of initial phase matrices
    error(['Requested ', num2str(num_sources), ' encoding matrices. Given ', num2str(length(config.P_init)), ' initial phase matrices.']);
elseif ~iscell(config.P_init)  % given a matrix
    is_P_cell = false;
    config.P_init = {config.P_init};
else  % organize phase matrices as {P_1; P_2; ...; P_num_sources}
    is_P_cell = true;
    config.P_init = config.P_init(:);
end

% Update switches for phase matrices
if ~isfield(config, 'P_fixed') || isempty(config.P_fixed)  % not given an update switch. Fill this in.
    config.P_fixed = cell(num_sources, 1);
    for i = 1 : num_sources
        config.P_fixed{i} = false;
    end
elseif iscell(config.P_fixed) && length(config.P_fixed) > 1 && length(config.P_fixed) ~= num_sources  % given an incorrect number of update switches
    error(['Requested ', num2str(num_sources), ' basis matrices. Given ', num2str(length(config.P_fixed)), ' update switches.']);
elseif ~iscell(config.P_fixed)  || length(config.P_fixed) == 1  % extend one update switch level to all phase matrices
    if iscell(config.P_fixed)
        temp = config.P_fixed{1};
    else
        temp = config.P_fixed;
    end
    config.P_fixed = cell(num_sources, 1);
    for i = 1 : num_sources
        config.P_fixed{i} = temp;
    end
    clear temp;
end

W = config.W_init;
for i = 1 : num_sources
    W{i} = W{i} * diag(1 ./ sqrt(sum(W{i}.^2, 1)));
end

H = config.H_init;

W_all = cell2mat(W);
H_all = cell2mat(H);

P = config.P_init;
beta = cell(num_sources, 1);
V_hat_per_source = zeros(m, n, num_sources);
for i = 1 : num_sources
    V_hat_per_source(:, :, i) = (W{i} * H{i}) .* P{i};
end

V_hat = sum(V_hat_per_source, 3);

V_bar_per_source = zeros(m, n, num_sources);

cost = zeros(config.maxiter, 1);

for iter = 1 : config.maxiter
    % Update auxiliary variables
    for i = 1 : num_sources
        beta{i} = (W{i} * H{i}) ./ (W_all * H_all);
        V_bar_per_source(:, :, i) = V_hat_per_source(:, :, i) + beta{i} .* (V - V_hat);
    end
    
    % Update phase matrices
    for i = 1 : num_sources
        if ~config.P_fixed{i}
            P{i} = exp(1j * angle(V_bar_per_source(:, :, i))); %V_bar_per_source(:, :, i) ./ abs(V_bar_per_source(:, :, i));
        end
    end
    
    % Update basis matrices
    for i = 1 : num_sources
        if ~config.W_fixed{i}
            W{i} = W{i} .* (((abs(V_bar_per_source(:, :, i)) ./ beta{i}) * H{i}') ./ max(W_all * H_all * H{i}', eps));
            W{i} = W{i} * diag(1 ./ sqrt(sum(W{i}.^2, 1)));
        end
    end
    
    % Update encoding matrices
    for i = 1 : num_sources
        if ~config.H_fixed{i}
            H{i} = H{i} .* ((W{i}' * (abs(V_bar_per_source(:, :, i)) ./ beta{i})) ./ max(W{i}' * W_all * H_all + config.H_sparsity{i}, eps)); %max(H .* ((W.^2)' * (ones(m, n) ./ beta)) + config.H_sparsity, eps));
        end
    end
    
    W_all = cell2mat(W);
    H_all = cell2mat(H);
    
    for i = 1 : num_sources
        V_hat_per_source(:, :, i) = (W{i} * H{i}) .* P{i};
    end

    V_hat = sum(V_hat_per_source, 3);
    
    % Calculate cost for this iteration
    cost(iter) = sum(sum(abs(V - V_hat).^2));
    for i = 1 : num_sources
        cost(iter) = cost(iter) + config.H_sparsity{i} * sum(sum(H{i}));
    end
    
    % Stop iterations if change in cost function less than the tolerance
    if iter > 1 && cost(iter) < cost(iter-1) && cost(iter-1) - cost(iter) < config.tolerance
        cost = cost(1 : iter);  % trim vector
        break;
    end
end

% Prepare the output
if ~is_W_cell
    W = W{1};
end

if ~is_H_cell
    H = H{1};
end

if ~is_P_cell
    P = P{1};
end
