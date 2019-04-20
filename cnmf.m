function [W, H, cost] = cnmf(V, num_basis_elems, context_len, config)
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
%   context_len: [positive scalar]
%       number of context frames to use when learning the factorization.
%       1 = NMF.
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
%           m-by-num_basis_elems-by-context_len tensor.
%           [cell array]: initialize K basis tensors with a K-length array
%           containing {m-by-num_basis_elems_1-by-context_len, ...,
%           m-by-num_basis_elems_K-by-context_len} non-negative tensors. 
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
%       [non-negative 3D tensor]: m-by-num_basis_elems-by-context_len
%       non-negative basis tensor.
%       [cell array]: K-length array containing {m-by-num_basis_elems_1-
%       by-context_len, ..., m-by-num_basis_elems_K-by-context_len}
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
[config, is_W_cell, is_H_cell] = ValidateParameters(V, num_basis_elems, context_len, config);

if (strcmp(config.divergence, 'ab_divergence') || strcmp(config.divergence, 'ab')) && config.alpha == 0 && config.beta == 0
    error('alpha = 0 and beta = 0 is not supported at this time.');
end

switch config.divergence
    case {'euclidean', 'frobenius'}
        config.alpha = 1;
        config.beta = 1;
    case {'kl_divergence', 'kl'}
        config.alpha = 1;
        config.beta = 0;
    case {'is_divergence', 'is'}
        config.alpha = 1;
        config.beta = -1;
end

if config.alpha == 0  % when alpha is 0, need to use the dual update equations for AB-divergence
    use_dual = true;
else
    use_dual = false;
end

W = config.W_init;
H = config.H_init;
for source_count = 1 : num_sources
%     for t = 1 : context_len
%         W{source_count}(:, :, t) = W{source_count}(:, :, t) * diag(1 ./ sqrt(sum(W{source_count}(:, :, t).^2, 1)));
%     end
    for k = 1 : num_basis_elems{source_count}
        w_norm = norm(squeeze(W{source_count}(:, k, :)), 'fro') / context_len;
        W{source_count}(:, k, :) = W{source_count}(:, k, :) / w_norm;
        H{source_count}(k, :) = w_norm * H{source_count}(k, :);
    end
end

W_all = cell2mat(W);
H_all = cell2mat(H);

V_hat = ReconstructFromDecomposition(W_all, H_all);

cost = zeros(config.maxiter, 1);

for iter = 1 : config.maxiter
    % Update basis matrices
    for source_count = 1 : num_sources
        if ~config.W_fixed{source_count}
            if use_dual
                for t = 1 : context_len
                    H_shifted = [zeros(num_basis_elems{source_count}, t-1) H{source_count}(:, 1:n-t+1)];
                    gradient_neg = ((V.^(config.alpha - 1) .* V_hat.^config.beta) * H_shifted' + W{source_count}(:, :, t) * diag(diag(H_shifted * V'.^(config.alpha + config.beta - 1) * W{source_count}(:, :, t)))).^(1 / config.beta);
                    gradient_pos = (V.^(config.alpha + config.beta - 1) * H_shifted' + W{source_count}(:, :, t) * diag(diag(H_shifted * (V.^(config.alpha - 1) .* V_hat.^config.beta)' * W{source_count}(:, :, t)))).^(1 / config.beta);
                    W{source_count}(:, :, t) = W{source_count}(:, :, t) .* (gradient_neg ./ max(gradient_pos + config.W_sparsity{source_count}, eps));
                end
            else
                for t = 1 : context_len
                    H_shifted = [zeros(num_basis_elems{source_count}, t-1) H{source_count}(:, 1:n-t+1)];
%                     gradient_neg = ((V.^config.alpha .* V_hat.^(config.beta - 1)) * H_shifted').^(1 / config.alpha);
%                     gradient_pos = (V_hat.^(config.alpha + config.beta - 1) * H_shifted').^(1 / config.alpha);
                    gradient_neg = ((V.^config.alpha .* V_hat.^(config.beta - 1)) * H_shifted' + W{source_count}(:, :, t) * diag(diag(H_shifted * V_hat'.^(config.alpha + config.beta - 1) * W{source_count}(:, :, t)))).^(1 / config.alpha);
                    gradient_pos = (V_hat.^(config.alpha + config.beta - 1) * H_shifted' + W{source_count}(:, :, t) * diag(diag(H_shifted * (V.^config.alpha .* V_hat.^(config.beta - 1))' * W{source_count}(:, :, t)))).^(1 / config.alpha);
                    W{source_count}(:, :, t) = W{source_count}(:, :, t) .* (gradient_neg ./ max(gradient_pos + config.W_sparsity{source_count}, eps));
                end
            end
            for k = 1 : num_basis_elems{source_count}
                w_norm = norm(squeeze(W{source_count}(:, k, :)), 'fro') / context_len;
                W{source_count}(:, k, :) = W{source_count}(:, k, :) / w_norm;
            end
        end
    end
    W_all = cell2mat(W);
    H_all = cell2mat(H);
    V_hat = ReconstructFromDecomposition(W_all, H_all);
    
    % Update encoding matrices
    for source_count = 1 : num_sources
        if ~config.H_fixed{source_count}
            if use_dual
                V_neg = V.^(config.alpha - 1) .* V_hat.^config.beta;
                V_pos = V.^(config.alpha + config.beta - 1);
            else
                V_neg = V.^config.alpha .* V_hat.^(config.beta - 1);
                V_pos = V_hat.^(config.alpha + config.beta - 1);
            end
            gradient_neg = zeros(num_basis_elems{source_count}, n);
            gradient_pos = zeros(num_basis_elems{source_count}, n);
            for t = 1 : context_len
                V_neg_shifted = [V_neg(:, t:n) zeros(m, t-1)];
                if strcmp(config.divergence, 'kl_divergence') || strcmp(config.divergence, 'kl')  % TODO: check if this needs to be done for other divergences
                    V_pos_shifted = V_pos;
                else
                    V_pos_shifted = [V_pos(:, t:n) zeros(m, t-1)];
                end
                gradient_neg = gradient_neg + W{source_count}(:, :, t)' * V_neg_shifted;
                gradient_pos = gradient_pos + W{source_count}(:, :, t)' * V_pos_shifted;
            end
            if use_dual
                H{source_count} = H{source_count} .* (gradient_neg.^(1 / config.beta) ./ max(gradient_pos.^(1 / config.beta) + config.H_sparsity{source_count}, eps));
            else
                H{source_count} = H{source_count} .* (gradient_neg.^(1 / config.alpha) ./ max(gradient_pos.^(1 / config.alpha) + config.H_sparsity{source_count}, eps));
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
    for source_count = 1 : num_sources
        cost(iter) = cost(iter) + config.W_sparsity{source_count} * sum(sum(sum(abs(W{source_count})))) + config.H_sparsity{source_count} * sum(sum(abs(H{source_count})));
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

end  % function cnmf

function [config_out, is_W_cell, is_H_cell] = ValidateParameters(V, num_basis_elems, context_len, config_in)
% ValdidateParameters private function
% Check parameters supplied by the user and fill in default values for
% unsupplied parameters.

config_out = config_in;

[data_dim, num_samples] = size(V);

num_sources = length(num_basis_elems);

% Initialize default divergence for the cost function
if ~isfield(config_out, 'divergence')
    config_out.divergence = 'euclidean';
end

% Default alpha parameter for Alpha-Beta divergence
if ~isfield(config_out, 'alpha')
    config_out.alpha = 1;
elseif ~strcmp(config_out.divergence, 'ab_divergence') && ~strcmp(config_out.divergence, 'ab')
    config_out.alpha = 1;
end

% Default beta parameter for Alpha-Beta divergence
if ~isfield(config_out, 'beta')
    config_out.beta = 1;
elseif ~strcmp(config_out.divergence, 'ab_divergence') && ~strcmp(config_out.divergence, 'ab')
    config_out.beta = 1;
end

% Initialize encoding matrices
if ~isfield(config_out, 'H_init') || isempty(config_out.H_init)  % not given any inital encoding matrices. Fill these in.
    if num_sources == 1
        is_H_cell = false;
    else
        is_H_cell = true;
    end
    config_out.H_init = cell(num_sources, 1);
    for source_count = 1 : num_sources
        config_out.H_init{source_count} = max(rand(num_basis_elems{source_count}, num_samples), eps);
    end
elseif iscell(config_out.H_init) && length(config_out.H_init) ~= num_sources  % given an incorrect number of initial encoding matrices
    error(['Requested ', num2str(num_sources), ' sources. Given ', num2str(length(config_out.H_init)), ' initial encoding matrices.']);
elseif ~iscell(config_out.H_init)  % given a matrix
    is_H_cell = false;
    config_out.H_init = {config_out.H_init};
else  % organize encoding matrices as {H_1; H_2; ...; H_num_bases}
    is_H_cell = true;
    config_out.H_init = config_out.H_init(:);
end

% Initialize basis matrices
if ~isfield(config_out, 'W_init') || isempty(config_out.W_init)  % not given any inital basis matrices. Fill these in.
    if num_sources == 1
        is_W_cell = false;
    else
        is_W_cell = true;
    end
    config_out.W_init = cell(1, num_sources);
    for source_count = 1 : num_sources
        config_out.W_init{source_count} = rand(data_dim, num_basis_elems{source_count}, context_len);
        for k = 1 : num_basis_elems{source_count}
            w_norm = norm(squeeze(config_out.W_init{source_count}(:, k, :)), 'fro') / context_len;
            config_out.W_init{source_count}(:, k, :) = config_out.W_init{source_count}(:, k, :) / w_norm;
        end
    end
elseif iscell(config_out.W_init) && length(config_out.W_init) ~= num_sources  % given an incorrect number of initial basis matrices
    error(['Requested ', num2str(num_sources), ' sources. Given ', num2str(length(config_out.W_init)), ' initial basis matrices.']);
elseif ~iscell(config_out.W_init)  % given a matrix
    is_W_cell = false;
    config_out.W_init = {config_out.W_init};
else  % organize basis matrices as {W_1 W_2 ... W_num_bases}
    is_W_cell = true;
    config_out.W_init = config_out.W_init(:)';
end

% Sparsity levels for basis matrices
if ~isfield(config_out, 'W_sparsity') || isempty(config_out.W_sparsity)  % not given a sparsity level. Fill this in.
    config_out.W_sparsity = cell(num_sources, 1);
    for source_count = 1 : num_sources
        config_out.W_sparsity{source_count} = 0;
    end
elseif iscell(config_out.W_sparsity) && length(config_out.W_sparsity) > 1 && length(config_out.W_sparsity) ~= num_sources  % given an incorrect number of sparsity levels
    error(['Requested ', num2str(num_sources), ' sources. Given ', num2str(length(config_out.W_sparsity)), ' sparsity levels.']);
elseif ~iscell(config_out.W_sparsity)  || length(config_out.W_sparsity) == 1  % extend one sparsity level to all basis matrices
    if iscell(config_out.W_sparsity)
        temp = max(config_out.W_sparsity{1}, 0);
    else
        temp = max(config_out.W_sparsity, 0);
    end
    config_out.W_sparsity = cell(num_sources, 1);
    for source_count = 1 : num_sources
        config_out.W_sparsity{source_count} = temp;
    end
    clear temp;
else  % make sure all given sparsity levels are non-negative
    for source_count = 1 : num_sources
        config_out.W_sparsity{source_count} = max(config_out.W_sparsity{source_count}, 0);
    end
end

% Sparsity levels for encoding matrices
if ~isfield(config_out, 'H_sparsity') || isempty(config_out.H_sparsity)  % not given a sparsity level. Fill this in.
    config_out.H_sparsity = cell(num_sources, 1);
    for source_count = 1 : num_sources
        config_out.H_sparsity{source_count} = 0;
    end
elseif iscell(config_out.H_sparsity) && length(config_out.H_sparsity) > 1 && length(config_out.H_sparsity) ~= num_sources  % given an incorrect number of sparsity levels
    error(['Requested ', num2str(num_sources), ' sources. Given ', num2str(length(config_out.H_sparsity)), ' sparsity levels.']);
elseif ~iscell(config_out.H_sparsity)  || length(config_out.H_sparsity) == 1  % extend one sparsity level to all encoding matrices
    if iscell(config_out.H_sparsity)
        temp = max(config_out.H_sparsity{1}, 0);
    else
        temp = max(config_out.H_sparsity, 0);
    end
    config_out.H_sparsity = cell(num_sources, 1);
    for source_count = 1 : num_sources
        config_out.H_sparsity{source_count} = temp;
    end
    clear temp;
else  % make sure all given sparsity levels are non-negative
    for source_count = 1 : num_sources
        config_out.H_sparsity{source_count} = max(config_out.H_sparsity{source_count}, 0);
    end
end

% Update switches for basis matrices
if ~isfield(config_out, 'W_fixed') || isempty(config_out.W_fixed)  % not given an update switch. Fill this in.
    config_out.W_fixed = cell(num_sources, 1);
    for source_count = 1 : num_sources
        config_out.W_fixed{source_count} = false;
    end
elseif iscell(config_out.W_fixed) && length(config_out.W_fixed) > 1 && length(config_out.W_fixed) ~= num_sources  % given an incorrect number of update switches
    error(['Requested ', num2str(num_sources), ' sources. Given ', num2str(length(config_out.W_fixed)), ' update switches.']);
elseif ~iscell(config_out.W_fixed)  || length(config_out.W_fixed) == 1  % extend one update switch level to all basis matrices
    if iscell(config_out.W_fixed)
        temp = config_out.W_fixed{1};
    else
        temp = config_out.W_fixed;
    end
    config_out.W_fixed = cell(num_sources, 1);
    for source_count = 1 : num_sources
        config_out.W_fixed{source_count} = temp;
    end
    clear temp;
end

% Update switches for encoding matrices
if ~isfield(config_out, 'H_fixed') || isempty(config_out.H_fixed)  % not given an update switch. Fill this in.
    config_out.H_fixed = cell(num_sources, 1);
    for source_count = 1 : num_sources
        config_out.H_fixed{source_count} = false;
    end
elseif iscell(config_out.H_fixed) && length(config_out.H_fixed) > 1 && length(config_out.H_fixed) ~= num_sources  % given an incorrect number of update switches
    error(['Requested ', num2str(num_sources), ' sources. Given ', num2str(length(config_out.H_fixed)), ' update switches.']);
elseif ~iscell(config_out.H_fixed)  || length(config_out.H_fixed) == 1  % extend one update switch level to all encoding matrices
    if iscell(config_out.H_fixed)
        temp = config_out.H_fixed{1};
    else
        temp = config_out.H_fixed;
    end
    config_out.H_fixed = cell(num_sources, 1);
    for source_count = 1 : num_sources
        config_out.H_fixed{source_count} = temp;
    end
    clear temp;
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
