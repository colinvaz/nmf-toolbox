function [W, H, cost] = nmf(V, num_basis_elems, config)
% nmf Decompose a non-negative matrix V into WH using NMF [1] by minimizing
% a chosen divergence. W is a basis matrix and H is the encoding matrix that
% encodes the input V in terms of the basis W. This function can output
% multiple basis/encoding matrices for multiple sources, each of which can 
% be fixed to a given matrix or have a given sparsity level.
%
% Inputs:
%   V: [non-negative matrix]
%       m-by-n non-negative matrix containing data to be decomposed.
%   num_basis_elems: [positive scalar] or [cell array]
%       [positive scalar]: number of basis elements (columns of W/rows of H)
%       for 1 source.
%       [cell array]: K-length array of positive scalars {num_basis_elems_1,
%       ...,num_basis_elems_K} specifying the number of basis elements for
%       K sources.
%   config: [structure] (optional)
%       structure containing configuration parameters.
%       config.divergence: [string] (default: 'euclidean')
%           divergence metric of the cost function. Can be 'euclidean',
%           'kl_divergence', 'kl', 'is_divergence', 'is', 'ab_divergence',
%           or 'ab'.
%       config.alpha: [scalar] (default: 1)
%           alpha parameter for the AB-divergence. Used when
%           config.divergence is 'ab_divergence' or 'ab'.
%       config.beta: [scalar] (default: 1)
%           beta parameter for the AB-divergence. Used when
%           config.divergence is 'ab_divergence' or 'ab'. (default: 1)
%       config.W_init: [non-negative matrix] or [cell array] (default:
%           random matrix or K-length cell array of random matrices)
%           [non-negative matrix]: initialize 1 basis matrix for 1 source
%           with a m-by-num_basis_elems matrix.
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
%   cost: [vector]
%       value of the cost function after each iteration.
%
% The AB-divergence generalizes the alpha and beta families of divergences,
% including the Euclidean distance, KL divergence, and IS divergence.
% Setting alpha and beta (both real-valued) to different values achieves
% these divergences and more. See [2].
% alpha and beta values for common divergence metrics:
%   alpha       beta        divergence
%   -----       ----        ----------
%   1           1           Scaled squared Euclidean
%   1           0           Kullback-Leibler KLdiv(V || V_hat)
%   0           1           Kullback-Leibler KLdiv(V_hat || V)
%   1           -1          Itakura-Saito
%   0.5         0.5         Hellinger
%   alpha       1-alpha     Alpha divergences
%   1           beta        Beta divergences
% Currently, there are no update equations when alpha = 0 and beta = 0.
%
% References:
%   [1] D. D. Lee and H. S. Seung, "Algorithms for non-negative matrix
%       factorization," in NIPS, Denver, CO, 2001, pp. 556-562.
%   [2] A. Cichocki, S. Cruces, and S. Amari, "Generalized Alpha-Beta
%       Divergences and Their Application to Robust Nonnegative Matrix
%       Factorization," Entropy, vol. 13, no. 1, pp. 134-170, Jan. 2011.
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
if ~iscell(num_basis_elems)
    num_basis_elems = {num_basis_elems};
end
num_sources = length(num_basis_elems);
[config, is_W_cell, is_H_cell] = ValidateParameters('nmf', config, V, num_basis_elems);

if ~isfield(config, 'divergence')
    config.divergence = 'euclidean';
end
if config.alpha == 0  % when alpha is 0, need to use the dual update equations for AB-divergence
    use_dual = true;
else
    use_dual = false;
end

if (strcmp(config.divergence, 'ab_divergence') || strcmp(config.divergence, 'ab')) && config.alpha == 0 && config.beta == 0
    error('alpha = 0 and beta = 0 is not supported at this time.');
end

W = config.W_init;
H = config.H_init;
for i = 1 : num_sources
    W{i} = W{i} * diag(1 ./ sqrt(sum(W{i}.^2, 1)));
end

W_all = cell2mat(W);
H_all = cell2mat(H);

V_hat = ReconstructFromDecomposition(W_all, H_all);

cost = zeros(config.maxiter, 1);

for iter = 1 : config.maxiter
    % Update basis matrices
    for i = 1 : num_sources
        if ~config.W_fixed{i}
            switch config.divergence
                case 'euclidean'
                    negative_grad = V * H{i}' + W{i} * diag(diag(H{i} * V_hat' * W{i}));
                    positive_grad = V_hat * H{i}' + W{i} * diag(diag(H{i} * V' * W{i}));
                case {'kl_divergence', 'kl'}
                    negative_grad = (V ./ V_hat) * H{i}' + W{i} * diag(diag(H{i} * ones(n, m) * W{i}));
                    positive_grad = ones(m, n) * H{i}' + W{i} * diag(diag(H{i} * (V' ./ V_hat') * W{i}));
                case {'is_divergence', 'is'}
                    negative_grad = (V ./ V_hat.^2) * H{i}' + W{i} * diag(diag(H{i} * (ones(n, m) ./ V_hat') * W{i}));
                    positive_grad = (ones(m, n) ./ V_hat) * H{i}' + W{i} * diag(diag(H{i} * (V' ./ V_hat'.^2) * W{i}));
                case {'ab_divergence', 'ab'}
                    if use_dual
                        negative_grad = ((V.^(config.alpha - 1) .* V_hat.^config.beta) * H{i}' + W{i} * diag(diag(H{i} * V'.^(config.alpha + config.beta - 1) * W{i}))).^(1 / config.beta);
                        positive_grad = (V.^(config.alpha + config.beta - 1) * H{i}' + W{i} * diag(diag(H{i} * (V.^(config.alpha - 1) .* V_hat.^config.beta)' * W{i}))).^(1 / config.beta);
                    else
                        negative_grad = ((V.^config.alpha .* V_hat.^(config.beta - 1)) * H{i}' + W{i} * diag(diag(H{i} * V_hat'.^(config.alpha + config.beta - 1) * W{i}))).^(1 / config.alpha);
                        positive_grad = (V_hat.^(config.alpha + config.beta - 1) * H{i}' + W{i} * diag(diag(H{i} * (V.^config.alpha .* V_hat.^(config.beta - 1))' * W{i}))).^(1 / config.alpha);
                    end
                otherwise
                    error(['No update equations defined for cost function with divergence type ', config.divergence]);
            end
            W{i} = W{i} .* (negative_grad ./ max(positive_grad + config.W_sparsity{i}, eps));
            W{i} = W{i} * diag(1 ./ sqrt(sum(W{i}.^2, 1)));
        end
    end
    W_all = cell2mat(W);
    V_hat = ReconstructFromDecomposition(W_all, H_all);
    
    % Update encoding matrices
    for i = 1 : num_sources
        if ~config.H_fixed{i}
            switch config.divergence
                case 'euclidean'
                    negative_grad = W{i}' * V;
                    positive_grad = W{i}' * V_hat;
                case {'kl_divergence', 'kl'}
                    negative_grad = W{i}' * (V ./ V_hat);
                    positive_grad = W{i}' * ones(m, n);
                case {'is_divergence', 'is'}
                    negative_grad = W{i}' * (V ./ V_hat.^2);
                    positive_grad = W{i}' * (ones(m, n) ./ (W{i} * H{i}));
                case {'ab_divergence', 'ab'}
                    if use_dual
                        negative_grad = (W{i}' * (V.^(config.alpha - 1) .* V_hat.^config.beta)).^(1 / config.beta);
                        positive_grad = (W{i}' * V.^(config.alpha + config.beta - 1)).^(1 / config.beta);
                    else
                        negative_grad = (W{i}' * V.^config.alpha .* V_hat.^(config.beta - 1)).^(1 / config.alpha);
                        positive_grad = (W{i}' * V_hat.^(config.alpha + config.beta - 1)).^(1 / config.alpha);
                    end
                otherwise
                    error(['No update equations defined for cost function with divergence type ', config.divergence]);
            end
            H{i} = H{i} .* (negative_grad ./ max(positive_grad + config.H_sparsity{i}, eps));
        end
    end
    H_all = cell2mat(H);
    V_hat = ReconstructFromDecomposition(W_all, H_all);
    
    % Calculate cost for this iteration
    switch config.divergence
        case 'euclidean'
            cost(iter) = 0.5 * sum(sum((V - V_hat).^2));
        case {'kl_divergence', 'kl'}
            cost(iter) = sum(sum(V .* log(V ./ V_hat) - V + V_hat));
        case {'is_divergence', 'is'}
            cost(iter) = sum(sum(log(V_hat ./ V) + (V ./ V_hat) - 1));
        case {'ab_divergence', 'ab'}
            cost(iter) = (-1 / (config.alpha * config.beta)) * sum(sum(V.^config.alpha .* V_hat.^config.beta - (config.alpha * V.^(config.alpha + config.beta) + config.beta *  V_hat.^(config.alpha + config.beta) + config.beta) / (config.alpha + config.beta)));
    end
    for i = 1 : num_sources
        cost(iter) = cost(iter) + config.W_sparsity{i} * sum(sum(abs(W{i}))) + config.H_sparsity{i} * sum(sum(abs(H{i})));
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
