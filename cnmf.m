function [W, H, cost] = cnmf(V, num_basis_elems, num_frames, config)
% cnmf Decompose a non-negative matrix V into WH using CNMF [2] by minimizing
% a chosen divergence. W is a time-varying basis tensor and H is the encoding
% matrix that encodes the input V in terms of the basis W. This function
% can output multiple basis tensors/encoding matrices for multiple sources,
% each of which can be fixed to a given tensor/matrix or have a given
% sparsity level.
%
% Inputs:
%   V: [non-negative matrix]
%       m-by-n matrix containing data to be decomposed.
%                 ----------------
%         data    |              |
%       dimension |      V       |
%                 |              |
%                 ----------------
%                  time --->
%   num_basis_elems: [positive scalar] or [cell array]
%       [positive scalar]: number of basis elements (columns of W/rows of H)
%       for 1 source.
%       [cell array]: K-length array of positive scalars {num_basis_elems_1,
%       ...,num_basis_elems_K} specifying the number of basis elements for
%       K sources.
%   num_frames: [positive scalar]
%       number of context frames. 1 = NMF.
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
%       config.W_init: [non-negative 3D tensor] or [cell array] (default:
%           random tensor or K-length cell array of random tensors)
%           [non-negative 3D tensor]: initialize 1 basis tensor with a
%           m-by-num_basis_elems-by-num_frames tensor.
%           [cell array]: initialize K basis tensors with a K-length array
%           containing {m-by-num_basis_elems_1-by-num_frames, ...,
%           m-by-num_basis_elems_K-by-num_frames} non-negative tensors. 
%       config.H_init: [non-negative matrix] or [cell array] (default:
%           random matrix or K-length cell array of random matrices)
%           [non-negative matrix]: initialize 1 encoding matrix with a
%           num_basis_elems-by-n non-negative matrix.
%           [cell array]: initialize K encoding matrices with a K-length
%           array containing {num_basis_elems_1-by-n, ...,
%           num_basis_elems_K-by-n} non-negative matrices. 
%       config.W_sparsity: [non-negative scalar] or [cell array] (default: 0)
%           [non-negative scalar]: sparsity level for all basis tensors.
%           [cell array]: K-length array of non-negative scalars indicating
%           the sparsity level of the K basis tensors.
%       config.H_sparsity: [non-negative scalar] or [cell array] (default: 0)
%           [non-negative scalar]: sparsity level for all K encoding matrices.
%           [cell array]: K-length array of non-negative scalars indicating
%           the sparsity level of the K encoding matrices.
%       config.W_fixed: [boolean] or [cell array] (default: false)
%           [boolean]: indicate if all basis tensors are fixed during the
%           update equations.
%           [cell array] K-length array indicating if the corresponding
%           basis tensors are fixed during the update equations.
%       config.H_fixed: [boolean] or [cell array] (default: false)
%           [boolean]: indicate if all encoding matrices are fixed during
%           the update equations.
%           [cell array] K-length array indicating if the corresponding
%           encoding matrices are fixed during the update equations.
%       config.maxiter: [positive scalar] (default: 100)
%           maximum number of update iterations.
%       config.tolerance: [positive scalar] (default: 1e-3)
%           maximum change in the cost function between iterations before
%           the algorithm is considered to have converged.
%
% Outputs:
%   W: [non-negative 3D tensor] or [cell array]
%       [non-negative 3D tensor]: m-by-num_basis_elems-by-num_frames
%       non-negative basis tensor.
%       [cell array]: K-length array containing {m-by-num_basis_elems_1-
%       by-num_frames, ..., m-by-num_basis_elems_K-by-num_frames}
%       non-negative basis tensors.
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
%   [1] P. Smaragdis, "Convolutive Speech Bases and their Application to
%       Speech Separation," IEEE Trans. Speech Audio Process., vol. 15,
%       no. 1, pp. 1-12, Jan. 2007.
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
if ~iscell(num_basis_elems)
    num_basis_elems = {num_basis_elems};
end
num_sources = length(num_basis_elems);
[config, is_W_cell, is_H_cell] = ValidateParameters('cnmf', config, V, num_basis_elems, num_frames);

if ~isfield(config, 'divergence')
    config.divergence = 'euclidean';
end
if config.alpha == 0  % when alpha is 0, need to use the dual update equations for AB-divergence
    use_dual = true;
else
    use_dual = false;
end

% Total Variance Denoising (TVD) switch for encoding matrices
if ~isfield(config, 'H_tvd') || isempty(config.H_tvd)  % not given whether to use TVD. Fill this in.
    config.H_tvd = cell(num_sources, 1);
    for i = 1 : num_sources
        config.H_tvd{i} = false;
    end
elseif iscell(config.H_tvd) && length(config.H_tvd) > 1 && length(config.H_tvd) ~= num_sources  % given an incorrect number of TVD switches
    error(['Requested ', num2str(num_sources), ' sources. Given ', num2str(length(config.H_tvd)), ' update switches.']);
elseif ~iscell(config.H_tvd)  || length(config.H_tvd) == 1  % extend one TVD switch to all encoding matrices
    if iscell(config.H_tvd)
        temp = config.H_tvd{1};
    else
        temp = config.H_tvd;
    end
    config.H_tvd = cell(num_sources, 1);
    for i = 1 : num_sources
        config.H_tvd{i} = temp;
    end
    clear temp;
end

% Initialize TVD alpha levels
if ~isfield(config, 'H_tvd_alpha') || isempty(config.H_tvd_alpha)  % not given an alpha level. Fill this in.
    config.H_tvd_alpha = cell(num_sources, 1);
    for i = 1 : num_sources
        config.H_tvd_alpha{i} = 1;
    end
elseif iscell(config.H_tvd_alpha) && length(config.H_tvd_alpha) > 1 && length(config.H_tvd_alpha) ~= num_sources  % given an incorrect number of alpha levels
    error(['Requested ', num2str(num_sources), ' sources. Given ', num2str(length(config.H_tvd_alpha)), ' TVD alpha levels.']);
elseif ~iscell(config.H_tvd_alpha)  || length(config.H_tvd_alpha) == 1  % extend one alpha level to all encoding matrices
    if iscell(config.H_tvd_alpha)
        temp = max(config.H_tvd_alpha{1}, 0);
    else
        temp = max(config.H_tvd_alpha, 0);
    end
    config.H_tvd_alpha = cell(num_sources, 1);
    for i = 1 : num_sources
        config.H_tvd_alpha{i} = temp;
    end
    clear temp;
else  % make sure all given alpha levels are non-negative
    for i = 1 : num_sources
        config.H_tvd_alpha{i} = max(config.H_tvd_alpha{i}, 0);
    end
end

if (strcmp(config.divergence, 'ab_divergence') || strcmp(config.divergence, 'ab')) && config.alpha == 0 && config.beta == 0
    error('alpha = 0 and beta = 0 is not supported at this time.');
end

W = config.W_init;
H = config.H_init;
for i = 1 : num_sources
%     for t = 1 : num_frames
%         W{i}(:, :, t) = W{i}(:, :, t) * diag(1 ./ sqrt(sum(W{i}(:, :, t).^2, 1)));
%     end
    for k = 1 : num_basis_elems{i}
        W_curr = permute(W{i}(:, k, :), [1 3 2]);
        W{i}(:, k, :) = permute(diag(1 ./ sqrt(sum(W_curr.^2, 2))) * W_curr, [1 3 2]);
    end
end

W_all = cell2mat(W);
H_all = cell2mat(H);

V_hat = ReconstructFromDecomposition(W_all, H_all);

cost = zeros(config.maxiter, 1);

for iter = 1 : config.maxiter
    % Update basis matrices
    for i = 1 : num_sources
        if ~config.W_fixed{i}
            W0 = W{i};
            gradient_neg = zeros(m, num_basis_elems{i}, num_frames);
            gradient_pos = zeros(m, num_basis_elems{i}, num_frames);
            switch config.divergence
                case 'euclidean'
                    for t = 1 : num_frames
                        H_shifted = [zeros(num_basis_elems{i}, t-1) H{i}(:, 1:n-t+1)];
%                         gradient_neg(:, :, t) = V * H_shifted' + W0(:, :, t) * diag(diag(H_shifted * V_hat' * W0(:, :, t)));
%                         gradient_pos(:, :, t) = V_hat * H_shifted' + W0(:, :, t) * diag(diag(H_shifted * V' * W0(:, :, t)));
                        gradient_neg(:, :, t) = V * H_shifted' + W0(:, :, t);
                        gradient_pos(:, :, t) = V_hat * H_shifted';
                        W{i}(:, :, t) = W0(:, :, t) .* (gradient_neg(:, :, t) ./ max(gradient_pos(:, :, t) + config.W_sparsity{i}, eps));
                        V_hat = max(V_hat + (W{i}(:, :, t) - W0(:, :, t)) * H_shifted, 0);
                    end
                case {'kl_divergence', 'kl'}
                    for t = 1 : num_frames
                        H_shifted = [zeros(num_basis_elems{i}, t-1) H{i}(:, 1:n-t+1)];
                        gradient_neg(:, :, t) = (V ./ V_hat) * H_shifted' + W0(:, :, t) * diag(diag(H_shifted * ones(n, m) * W0(:, :, t)));
                        gradient_pos(:, :, t) = ones(m, n) * H_shifted' + W0(:, :, t) * diag(diag(H_shifted * (V' ./ V_hat') * W0(:, :, t)));
                    end
                case {'is_divergence', 'is'}
                    for t = 1 : num_frames
                        H_shifted = [zeros(num_basis_elems{i}, t-1) H{i}(:, 1:n-t+1)];
                        gradient_neg(:, :, t) = (V ./ V_hat.^2) * H_shifted' + W0(:, :, t) * diag(diag(H_shifted * (ones(n, m) ./ V_hat') * W0(:, :, t)));
                        gradient_pos(:, :, t) = (ones(m, n) ./ V_hat) * H_shifted' + W0(:, :, t) * diag(diag(H_shifted * (V' ./ V_hat'.^2) * W0(:, :, t)));
                    end
                case {'ab_divergence', 'ab'}
                    if use_dual
                        for t = 1 : num_frames
                            H_shifted = [zeros(num_basis_elems{i}, t-1) H{i}(:, 1:n-t+1)];
                            gradient_neg(:, :, t) = ((V.^(config.alpha - 1) .* V_hat.^config.beta) * H_shifted' + W0(:, :, t) * diag(diag(H_shifted * V'.^(config.alpha + config.beta - 1) * W0(:, :, t)))).^(1 / config.beta);
                            gradient_pos(:, :, t) = (V.^(config.alpha + config.beta - 1) * H_shifted' + W0(:, :, t) * diag(diag(H_shifted * (V.^(config.alpha - 1) .* V_hat.^config.beta)' * W0(:, :, t)))).^(1 / config.beta);
                        end
                    else
                        for t = 1 : num_frames
                            H_shifted = [zeros(num_basis_elems{i}, t-1) H{i}(:, 1:n-t+1)];
                            gradient_neg(:, :, t) = ((V.^config.alpha .* V_hat.^(config.beta - 1)) * H_shifted' + W0(:, :, t) * diag(diag(H_shifted * V_hat'.^(config.alpha + config.beta - 1) * W0(:, :, t)))).^(1 / config.alpha);
                            gradient_pos(:, :, t) = (V_hat.^(config.alpha + config.beta - 1) * H_shifted' + W0(:, :, t) * diag(diag(H_shifted * (V.^config.alpha .* V_hat.^(config.beta - 1))' * W0(:, :, t)))).^(1 / config.alpha);
                        end
                    end
                otherwise
                    error(['No update equations defined for cost function with divergence type ', config.divergence]);
            end
            for t = 1 : num_frames
                W{i}(:, :, t) = W0(:, :, t) .* (gradient_neg(:, :, t) ./ max(gradient_pos(:, :, t) + config.W_sparsity{i}, eps));
%                 W{i}(:, :, t) = W{i}(:, :, t) * diag(1 ./ sqrt(sum(W{i}(:, :, t).^2, 1)));
%                 V_hat = max(V_hat + (W{i}(:, :, t) - W0(:, :, t)) * H_shifted, 0);
            end
            for k = 1 : num_basis_elems{i}
                W_curr = permute(W{i}(:, k, :), [1 3 2]);
                W{i}(:, k, :) = permute(diag(1 ./ sqrt(sum(W_curr.^2, 2))) * W_curr, [1 3 2]);
            end
        end
    end
    W_all = cell2mat(W);
    V_hat = ReconstructFromDecomposition(W_all, H_all);
    
    % Update encoding matrices
    for i = 1 : num_sources
        if ~config.H_fixed{i}
            switch config.divergence
                case 'euclidean'
                    V_neg = V;
                    V_pos = V_hat;
                case {'kl_divergence', 'kl'}
                    V_neg = V ./ V_hat;
                    V_pos = ones(m, n);
                case {'is_divergence', 'is'}
                    V_neg = V ./ V_hat.^2;
                    V_pos = ones(m, n) ./ V_hat;
                case {'ab_divergence', 'ab'}
                    if use_dual
                        V_neg = V.^(config.alpha - 1) .* V_hat.^config.beta;
                        V_pos = V.^(config.alpha + config.beta - 1);
                    else
                        V_neg = V.^config.alpha .* V_hat.^(config.beta - 1);
                        V_pos = V_hat.^(config.alpha + config.beta - 1);
                    end 
                otherwise
                    error(['No update equations defined for cost function with divergence type ', config.divergence]);
            end
            gradient_neg = zeros(num_basis_elems{i}, n);
            gradient_pos = zeros(num_basis_elems{i}, n);
            for t = 1 : num_frames
                V_neg_shifted = [V_neg(:, t:n) zeros(m, t-1)];
                if strcmp(config.divergence, 'kl_divergence') || strcmp(config.divergence, 'kl')  % TODO: check if this needs to be done for other divergences
                    V_pos_shifted = V_pos;
                else
                    V_pos_shifted = [V_pos(:, t:n) zeros(m, t-1)];
                end
                gradient_neg = gradient_neg + W{i}(:, :, t)' * V_neg_shifted;
                gradient_pos = gradient_pos + W{i}(:, :, t)' * V_pos_shifted;
            end
            if use_dual
                H{i} = H{i} .* (gradient_neg.^(1 / config.beta) ./ max(gradient_pos.^(1 / config.beta) + config.H_sparsity{i}, eps));
            else
                H{i} = H{i} .* (gradient_neg.^(1 / config.alpha) ./ max(gradient_pos.^(1 / config.alpha) + config.H_sparsity{i}, eps));
            end
            if config.H_tvd{i}
                for k = 1 : num_basis_elems{i}
                    H{i}(k, :) = TVD_mm(H{i}(k, :), config.H_tvd_alpha{i}, 100);
                end
            end
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
        cost(iter) = cost(iter) + config.H_sparsity{i} * sum(sum(abs(H{i})));
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
