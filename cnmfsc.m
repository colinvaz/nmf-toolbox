function [W, H, cost] = cnmfsc(V, num_basis_elems, context_len, config)
% cnmfsc Decompose a non-negative matrix V into WH using convolutive NMF
% with sparseness constraints [1]. W is a time-varying basis tensor and H
% is the encoding matrix that encodes the input V in terms of the basis W.
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
%   num_basis_elems: [positive scalar]
%       number of basis elements (columns of W/rows of H) for 1 source.
%   context_len: [positive scalar]
%       number of context frames to use when learning the factorization.
%       1 = NMF-SC.
%   config: [structure] (optional)
%       structure containing configuration parameters.
%       config.W_init: [non-negative 3D tensor] (default: random tensor)
%           initialize 1 basis tensor with a
%           m-by-num_basis_elems-by-context_len tensor.
%       config.H_init: [non-negative matrix] (default: random matrix)
%           initialize 1 encoding matrix with a num_basis_elems-by-n
%           non-negative matrix.
%       config.W_sparsity: [non-negative scalar] (default: 0)
%           sparsity level in [0, 1] for the basis tensor.
%       config.H_sparsity: [non-negative scalar] (default: 0)
%           sparsity level in [0, 1] for the encoding matrix.
%       config.W_fixed: [boolean] (default: false)
%           indicate if the basis tensor is fixed during the update
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
%   W: [non-negative 3D tensor]
%       m-by-num_basis_elems-by-context_len non-negative basis tensor.
%   H: [non-negative matrix]
%       num_basis_elems-by-n non-negative encoding matrix.
%   cost: [vector]
%       value of the cost function after each iteration.
%
% References:
%   [1] V. Ramanarayanan, L. Goldstein, and S. Narayanan, "Spatio-temporal
%       articulatory movement primitives during speech production
%       - extraction, interpretation, and validation," J. Acoustical
%       Society of America, vol. 134, no. 2, pp. 1378-1394, 2013.
%
% TODO: Add support for multiple sources
%       Add support for minimizing AB divergence
%
% NMF Toolbox
% Colin Vaz - cvaz@usc.edu
% Signal Analysis and Interpretation Lab (SAIL) - http://sail.usc.edu
% University of Southern California
% 2015
    
% Check that we have non-negative data
if min(V(:)) < 0
    error('Negative values in data!');
end
    
% Globally rescale data to avoid potential overflow/underflow
V = V / max(V(:));

% Data dimensions
[m, n] = size(V);

% Check if configuration structure is given
if nargin < 4
    config = struct;
end
    
% Initialize basis tensor
if ~isfield(config, 'W_init') || isempty(config.W_init)
    config.W_init = rand(m, num_basis_elems, context_len);
end

% Initialize encoding matrix
if ~isfield(config, 'H_init') || isempty(config.H_init)
    config.H_init = rand(num_basis_elems, n);
    config.H_init = diag(1 ./ sqrt(sum(config.H_init.^2, 2))) * config.H_init;
end

W0 = config.W_init;
W = W0;
H = config.H_init;

% Make initial basis have correct sparseness
if ~isfield(config, 'W_sparsity') || isempty(config.W_sparsity)
    config.W_sparsity = 0;
elseif config.W_sparsity > 0
    if config.W_sparsity > 1
        config.W_sparsity = 1;
    end
    L1a = sqrt(m) - (sqrt(m) - 1) * config.W_sparsity;
    for t = 1 : context_len
        for k = 1 : num_basis_elems
            W(:, k, t) = projfunc(W(:, k, t), L1a, 1, 1);
        end
    end
end

% Make initial encoding matrix have correct sparseness
if ~isfield(config, 'H_sparsity') || isempty(config.H_sparsity)
    config.H_sparsity = 0;
elseif config.H_sparsity > 0
    if config.H_sparsity > 1
        config.H_sparsity = 1;
    end
    L1s = sqrt(n) - (sqrt(n) - 1) * config.H_sparsity; 
    for k = 1 : num_basis_elems
        H(k, :) = (projfunc(H(k, :)', L1s, 1, 1))';
    end
end

% Update switch for basis tensor
if ~isfield(config, 'W_fixed') || isempty(config.W_fixed)
    config.W_fixed = false;
end

% Update switch for encoding matrix
if ~isfield(config, 'H_fixed') || isempty(config.H_fixed)
    config.H_fixed = false;
end

% Maximum number of update iterations
if ~isfield(config, 'maxiter') || config.maxiter <= 0
    config.maxiter = 100;
end

% Maximum tolerance in cost function change per iteration
if ~isfield(config, 'tolerance') || config.tolerance <= 0
    config.tolerance = 1e-3;
end

% Initial stepsizes
stepsizeW = ones(context_len, 1);
stepsizeH = 1;

% Calculate initial cost
cost = zeros(config.maxiter+1, 1);
V_hat = ReconstructFromDecomposition(W, H);
cost(1) = 0.5 * sum(sum((V - V_hat).^2));

for iter = 1 : config.maxiter
    % Update encoding matrix
    if ~config.H_fixed
        negative_grad = zeros(num_basis_elems, n);
        positive_grad = zeros(num_basis_elems, n);
        for t = 1 : context_len
            V_shifted = [V(:, t:n) zeros(m, t-1)];
            V_hat_shifted = [V_hat(:, t:n) zeros(m, t-1)];
            negative_grad = negative_grad + W0(:, :, t)' * V_shifted;
            positive_grad = positive_grad + W0(:, :, t)' * V_hat_shifted;
        end
        if config.H_sparsity > 0
            % Gradient for H
            dH = positive_grad - negative_grad;
            begobj = cost(iter);

            % Make sure we decrease the objective!
            while 1
                % Take step in direction of negative gradient, and project
                Hnew = H - stepsizeH * dH;
                for k = 1 : num_basis_elems
                    Hnew(k, :) = (projfunc(Hnew(k, :)', L1s, 1, 1))';
                end

                % Calculate new objective
                V_hat = ReconstructFromDecomposition(W0, Hnew);
                newobj = 0.5 * sum(sum((V - V_hat).^2));

                % If the objective decreased, we can continue...
                if newobj <= begobj
                    break;
                end

                % ...else decrease stepsize and try again
                stepsizeH = stepsizeH / 2;
                if stepsizeH < 1e-200
                    display('Algorithm converged');
                    cost = cost(1 : iter);  % trim
                    return; 
                end
            end

            % Slightly increase the stepsize
            stepsizeH = 1.2 * stepsizeH;
            H = Hnew;
        else
            % Update using standard CNMF multiplicative update rule
            H = H .* (negative_grad ./ (positive_grad + eps));

            % Renormalize so rows of H have constant energy
            norms = sqrt(sum(H.^2, 2))';
            H = diag(1 ./ norms) * H;
            for t = 1 : context_len
                W0(:, :, t) = W0(:, :, t) * diag(norms);
            end
        end
    end
    
    % Update basis matrix
    if ~config.W_fixed
        V_hat = ReconstructFromDecomposition(W0, H);
        if config.W_sparsity > 0
            for t = 1 : context_len
                begobj = 0.5 * sum(sum((V - V_hat).^2));

                % Gradient for W
                H_shifted = [zeros(num_basis_elems, t-1) H(:, 1:n-t+1)];
                negative_grad = V * H_shifted';
                positive_grad = V_hat * H_shifted';
                dW = positive_grad - negative_grad;

                % Make sure we decrease the objective!
                while 1
                    % Take step in direction of negative gradient, and project
                    Wnew = W0(:, :, t) - stepsizeW(t)*dW;
                    for k = 1 : num_basis_elems
                        Wnew(:, k) = projfunc(Wnew(:, k), L1a, 1, 1); 
                    end

                    % Calculate new objective
                    V_hat = ReconstructFromDecomposition(Wnew, H);
                    newobj = 0.5 * sum(sum((V - V_hat).^2));

                    % If the objective decreased, we can continue...
                    if newobj <= begobj
                        break;
                    end

                    % ...else decrease stepsize and try again
                    stepsizeW(t) = stepsizeW(t) / 2;
                    if stepsizeW(t) < 1e-200 
                        display('Algorithm converged');
                        cost = cost(1 : iter);  % trim
                        return; 
                    end
                end
                % Slightly increase the stepsize
                stepsizeW(t) = 1.2 * stepsizeW(t);
                W(:, :, t) = Wnew;
            end
        else
            % Update using standard CNMF multiplicative update rule
            for t = 1 : context_len
                H_shifted = [zeros(num_basis_elems, t-1) H(:, 1:n-t+1)];
                negative_grad = V * H_shifted';
                positive_grad = V_hat * H_shifted';
                W(:, :, t) = W0(:, :, t) .* (negative_grad ./ max(positive_grad, eps));
                V_hat = max(V_hat + (W(:, :, t) - W0(:, :, t)) * H_shifted, 0);
            end
        end
    end
    W0 = W;
        
    % Calculate cost for this iteration
    V_hat = ReconstructFromDecomposition(W0, H);
    cost(iter+1) = 0.5 * sum(sum((V - V_hat).^2));
    
    % Stop iterations if change in cost function less than the tolerance
    if iter > 1 && cost(iter+1) < cost(iter) && cost(iter) - cost(iter+1) < config.tolerance
        cost = cost(1 : iter+1);  % trim vector
        break;
    end
end
