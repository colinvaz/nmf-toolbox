function [W, H, Z, A, cost] = constrainednmf(V, labels, num_basis_elems, config)
% nmf Decompose a non-negative matrix V into WZA using Constrained NMF [1]
% by minimizing a chosen divergence. W is a basis matrix and H is the encoding matrix that
% encodes the input V in terms of the basis W. This function can output
% multiple basis/encoding matrices for multiple sources, each of which can 
% be fixed to a given matrix or have a given sparsity level.
%
% Inputs:
%   V: [non-negative matrix]
%       m-by-n non-negative matrix containing data to be decomposed. This
%       function assumes that the ordering of the samples does not matter
%       (i.e. do not use this function for ordered data, like time series)
%   labels: [vector]
%       n-length vector of class labels for the n samples in V. The class
%       labels should be non-negative integers. Any unlabeled samples
%       should be given a label of -1.
%   num_basis_elems: [positive scalar]
%       number of basis elements (columns of W/rows of H).
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
%       config.W_init: [non-negative matrix] or (default: random matrix)
%           initialize basis matrix with a m-by-num_basis_elems matrix.
%       config.W_sparsity: [non-negative scalar] (default: 0)
%           sparsity level for the basis matrix.
%       config.Z_sparsity: [non-negative scalar] (default: 0)
%           sparsity level for the cluster matrix.
%       config.W_fixed: [boolean] (default: false)
%           indicate if the basis matrix is fixed during the update
%           equations.
%       config.Z_fixed: [boolean] (default: false)
%           indicate if the cluster matrix is fixed during the update
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
%       num_basis_elems-by-n non-negative encoding matrix. H = Z * A
%   Z: [non-negative matrix]
%       num_basis_elems-by-(n+num_classes-num_labeled_samps) non-negative
%       cluster matrix.
%   A: [non-negative matrix]
%       (n+num_classes-num_labeled_samps)-by-n non-negative label matrix.
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
%   [1] H. Liu and Z. Wu, "Non-negative Matrix Factorization with
%       Constraints," in Proc. AAAI Conf. Artificial Intelligence, 2010.
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
if nargin < 4
	config = struct;
end

[m, n] = size(V);

assert(length(labels) == n, ['Length of the label vector not equal to number of samples. Length of label vector = ', num2str(length(labels)), '; number of samples = ', num2str(n)]);

if ~isfield(config, 'W_init') || isempty(config.W_init)
    config.W_init = rand(m, num_basis_elems);
end
if ~isfield(config, 'W_sparsity') || isempty(config.W_sparsity)
    config.W_sparsity = 0;
end
if ~isfield(config, 'Z_sparsity') || isempty(config.Z_sparsity)
    config.Z_sparsity = 0;
end
if ~isfield(config, 'W_fixed') || isempty(config.W_fixed)
    config.W_fixed = false;
end
if ~isfield(config, 'Z_fixed') || isempty(config.Z_fixed)
    config.Z_fixed = false;
end
if ~isfield(config, 'divergence')
    config.divergence = 'euclidean';
end
if ~isfield(config, 'alpha')
    config.alpha = 1;
elseif ~strcmp(config.divergence, 'ab_divergence') && ~strcmp(config.divergence, 'ab')
    config.alpha = 1;
end
if ~isfield(config, 'beta')
    config.beta = 1;
elseif ~strcmp(config.divergence, 'ab_divergence') && ~strcmp(config.divergence, 'ab')
    config.beta = 1;
end
if config.alpha == 0  % when alpha is 0, need to use the dual update equations for AB-divergence
    use_dual = true;
else
    use_dual = false;
end
if ~isfield(config, 'maxiter') || config.maxiter <= 0
    config.maxiter = 100;
end
if ~isfield(config, 'tolerance') || config.tolerance <= 0
    config.tolerance = 1e-3;
end

if (strcmp(config.divergence, 'ab_divergence') || strcmp(config.divergence, 'ab')) && config.alpha == 0 && config.beta == 0
    error('alpha = 0 and beta = 0 is not supported at this time.');
end

W = config.W_init;
W = W * diag(1 ./ sqrt(sum(W.^2, 1)));

% Preprocess labels so that the classes are consecutive positive integers
% from 1 to num classes. Unlabeled samples are given label -1
num_labeled_samps = length(find(labels > -1));
if num_labeled_samps < n  % some unlabeled samples
    [uniq_labels, ~, labels_processed] = unique(labels);  % labels_processed reassigns class labels to 1 to num_classes+1, with label -1 (unlabeled samples) being assigned to 1
    labels_processed = labels_processed - 1;  % make class labels from 0 to num_classes (original label -1 (unlabeled samples) now assigned to 0)
    labels_processed(labels_processed == 0) = -1;  % reset unlabeled samples to label -1
    num_classes = length(uniq_labels) - 1;
else
    [uniq_labels, ~, labels_processed] = unique(labels);  % labels_processed reassigns class labels to 1 to num_classes, with lowest label being assigned to 1
    num_classes = length(uniq_labels);
end

% Rearrange samples so that samples with the same label are contiguous.
% Note that we put the unlabeled samples at the beginning, rather than the
% end as in [1]
[sorted_labels, sorted_idx] = sort(labels_processed, 'ascend');
V = V(:, sorted_idx);

C = zeros(num_classes, num_labeled_samps);
for samp = n - num_labeled_samps + 1 : n
    C(sorted_labels(samp), samp - (n - num_labeled_samps)) = 1;
end
A = [eye(n - num_labeled_samps) zeros(n - num_labeled_samps, num_labeled_samps); zeros(num_classes, n - num_labeled_samps) C];
% Note that the identity matrix and C are swapped from what's in [1]
% because we put the unlabeled samples at the beginning rather than the end

Z = rand(num_basis_elems, n + num_classes - num_labeled_samps);
% Z = [rand(num_basis_elems, n - num_labeled_samps) eye(num_classes)+0.1*rand(num_classes, num_classes)];

H = Z * A;

V_hat = ReconstructFromDecomposition(W, H);

cost = zeros(config.maxiter, 1);

for iter = 1 : config.maxiter
    % Update basis matrix
    if ~config.W_fixed
        switch config.divergence
            case 'euclidean'
                negative_grad = V * H' + W * diag(diag(H * V_hat' * W));
                positive_grad = V_hat * H' + W * diag(diag(H * V' * W));
            case {'kl_divergence', 'kl'}
                negative_grad = (V ./ V_hat) * H' + W * diag(diag(H * ones(n, m) * W));
                positive_grad = ones(m, n) * H' + W * diag(diag(H * (V' ./ V_hat') * W));
            case {'is_divergence', 'is'}
                negative_grad = (V ./ V_hat.^2) * H' + W * diag(diag(H * (ones(n, m) ./ V_hat') * W));
                positive_grad = (ones(m, n) ./ V_hat) * H' + W * diag(diag(H * (V' ./ V_hat'.^2) * W));
            case {'ab_divergence', 'ab'}
                if use_dual
                    negative_grad = ((V.^(config.alpha - 1) .* V_hat.^config.beta) * H' + W * diag(diag(H * V'.^(config.alpha + config.beta - 1) * W))).^(1 / config.beta);
                    positive_grad = (V.^(config.alpha + config.beta - 1) * H' + W * diag(diag(H * (V.^(config.alpha - 1) .* V_hat.^config.beta)' * W))).^(1 / config.beta);
                else
                    negative_grad = ((V.^config.alpha .* V_hat.^(config.beta - 1)) * H' + W * diag(diag(H * V_hat'.^(config.alpha + config.beta - 1) * W))).^(1 / config.alpha);
                    positive_grad = (V_hat.^(config.alpha + config.beta - 1) * H' + W * diag(diag(H * (V.^config.alpha .* V_hat.^(config.beta - 1))' * W))).^(1 / config.alpha);
                end
            otherwise
                error(['No update equations defined for cost function with divergence type ', config.divergence]);
        end
        W = W .* (negative_grad ./ max(positive_grad + config.W_sparsity, eps));
        W = W * diag(1 ./ sqrt(sum(W.^2, 1)));
    end
    V_hat = ReconstructFromDecomposition(W, H);
    
    % Update cluster matrix
    if ~config.Z_fixed
        switch config.divergence
            case 'euclidean'
                negative_grad = W' * V * A';
                positive_grad = W' * V_hat * A';
            case {'kl_divergence', 'kl'}
                negative_grad = W' * (V ./ V_hat) * A';
                positive_grad = W' * ones(m, n) * A';
            case {'is_divergence', 'is'}
                negative_grad = W' * (V ./ V_hat.^2) * A';
                positive_grad = W' * (ones(m, n) ./ (W * H)) * A';
            case {'ab_divergence', 'ab'}
                if use_dual
                    negative_grad = (W' * (V.^(config.alpha - 1) .* V_hat.^config.beta) * A').^(1 / config.beta);
                    positive_grad = (W' * V.^(config.alpha + config.beta - 1) * A').^(1 / config.beta);
                else
                    negative_grad = (W' * V.^config.alpha .* V_hat.^(config.beta - 1) * A').^(1 / config.alpha);
                    positive_grad = (W' * V_hat.^(config.alpha + config.beta - 1) * A').^(1 / config.alpha);
                end
            otherwise
                error(['No update equations defined for cost function with divergence type ', config.divergence]);
        end
        Z = Z .* (negative_grad ./ max(positive_grad + config.Z_sparsity, eps));
    end
    H = Z * A;
    V_hat = ReconstructFromDecomposition(W, H);
    
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
    cost(iter) = cost(iter) + config.W_sparsity * sum(sum(abs(W))) + config.Z_sparsity * sum(sum(abs(Z)));
    
    % Stop iterations if change in cost function less than the tolerance
    if iter > 1 && cost(iter) < cost(iter-1) && cost(iter-1) - cost(iter) < config.tolerance
        cost = cost(1 : iter);  % trim vector
        break;
    end
end

% Prepare output
% V was reordered to deal with labels. Put A (and consequently H and V_hat
% in the original ordering of V.
A_temp = A;
for samp = 1 : n
    A(:, sorted_idx(samp)) = A_temp(:, samp);
end
H = Z * A;
