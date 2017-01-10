function [W, H, cost] = nmfsc(V, num_basis_elems, config)
% nmfsc Decompose a non-negative matrix V into WH using NMF with sparseness
% constraints [1]. W is a basis matrix and H is the encoding matrix that
% encodes the input V in terms of the basis W.
% 
% Inputs:
%   V: [non-negative matrix]
%       m-by-n matrix containing data to be decomposed.
%   num_basis_elems: [positive scalar]
%       number of basis elements (columns of W/rows of H) for 1 source.
%   config: [structure] (optional)
%       structure containing configuration parameters.
%       config.W_init: [non-negative matrix] (default: random matrix)
%           initialize 1 basis matrix with a m-by-num_basis_elems matrix.
%       config.H_init: [non-negative matrix] (default: random matrix)
%           initialize 1 encoding matrix with a num_basis_elems-by-n
%           non-negative matrix.
%       config.W_sparsity: [non-negative scalar] (default: 0)
%           sparsity level in [0, 1] for the basis matrix.
%       config.H_sparsity: [non-negative scalar] (default: 0)
%           sparsity level in [0, 1] for the encoding matrix.
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
%   [1] P. O. Hoyer, "Non-negative Matrix Factorization with Sparseness
%       Constraints," J. Machine Learning Research, vol. 5, pp. 1457-1469,
%       2004.
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
if nargin < 3
    config = struct;
end
    
% Initialize basis tensor
if ~isfield(config, 'W_init') || isempty(config.W_init)
    config.W_init = rand(m, num_basis_elems);
end

% Initialize encoding matrix
if ~isfield(config, 'H_init') || isempty(config.H_init)
    config.H_init = rand(num_basis_elems, n);
    config.H_init = diag(1 ./ sqrt(sum(config.H_init.^2, 2))) * config.H_init;
end

W = config.W_init;
H = config.H_init;

% Make initial basis have correct sparseness
if ~isfield(config, 'W_sparsity') || isempty(config.W_sparsity)
    config.W_sparsity = 0;
elseif config.W_sparsity > 0
    if config.W_sparsity > 1
        config.W_sparsity = 1;
    end
    L1a = sqrt(m) - (sqrt(m) - 1) * config.W_sparsity;
    for k = 1 : num_basis_elems
        W(:, k) = projfunc(W(:, k), L1a, 1, 1);
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
stepsizeW = 1;
stepsizeH = 1;

% Calculate initial cost
cost = zeros(config.maxiter+1, 1);
V_hat = ReconstructFromDecomposition(W, H);
cost(1) = 0.5 * sum(sum((V - V_hat).^2));

for iter = 1 : config.maxiter
    % Update encoding matrix
    if ~config.H_fixed
        negative_grad = W' * V;
        positive_grad = W' * V_hat;
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
                V_hat = ReconstructFromDecomposition(W, Hnew);
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
            H = H .* (negative_grad ./ max(positive_grad, eps));

            % Renormalize so rows of H have constant energy
            norms = sqrt(sum(H.^2, 2))';
            H = diag(1 ./ norms) * H;
            W = W * diag(norms);
        end
    end
    
    % Update basis matrix
    if ~config.W_fixed
        V_hat = ReconstructFromDecomposition(W, H);
        negative_grad = V * H';
        positive_grad = V_hat * H';
        if config.W_sparsity > 0
            begobj = 0.5 * sum(sum((V - V_hat).^2));

            % Gradient for W
            dW = positive_grad - negative_grad;

            % Make sure we decrease the objective!
            while 1
                % Take step in direction of negative gradient, and project
                Wnew = W - stepsizeW*dW;
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
                stepsizeW = stepsizeW / 2;
                if stepsizeW < 1e-200 
                    display('Algorithm converged');
                    cost = cost(1 : iter);  % trim
                    return; 
                end
            end
            % Slightly increase the stepsize
            stepsizeW = 1.2 * stepsizeW;
            W = Wnew;
        else
            % Update using standard NMF multiplicative update rule
            W = W .* (negative_grad ./ max(positive_grad, eps));
        end
    end
        
    % Calculate cost for this iteration
    V_hat = ReconstructFromDecomposition(W, H);
    cost(iter+1) = 0.5 * sum(sum((V - V_hat).^2));
    
    % Stop iterations if change in cost function less than the tolerance
    if iter > 1 && cost(iter+1) < cost(iter) && cost(iter) - cost(iter+1) < config.tolerance
        cost = cost(1 : iter+1);  % trim vector
        break;
    end
end
