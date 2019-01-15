function [W, H, S, G, cost] = chcnmf(V, num_basis_elems, context_len, config)
% chcnmf Decompose a matrix V into SGH using Convex hull-CNMF [1] by
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
%                 ----------------
%         data    |              |
%       dimension |      V       |
%                 |              |
%                 ----------------
%                  time --->
%   num_basis_elems: [positive scalar]
%       number of basis elements (columns of G/rows of H) for 1 source.
%   context_len: [positive scalar]
%       number of context frames to use when learning the factorization.
%       1 = CH-NMF
%   config: [structure] (optional)
%       structure containing configuration parameters.
%       config.S_init: [matrix] (default: matrix returned by Matlab's
%           convhull function with input V)
%           initialize m-by-p matrix containing p points belonging to the
%           convex hull of V. 
%       config.G_init: [non-negative 3D tensor] (default: random tensor)
%           initialize time-varying convex combination matrix with a
%           p-by-num_basis_elems-by-context_len tensor.
%       config.H_init: [non-negative matrix] (default: n indicator vectors
%           of cluster membership using K-means + 0.2)
%           initialize encoding matrix with a num_basis_elems-by-n
%           non-negative matrix.
%       config.G_fixed: [boolean] (default: false)
%           indicate if the time-varying convex combination matrix is fixed
%           during the update equations.
%       config.H_fixed: [boolean] (default: false)
%           indicate if the encoding matrix is fixed during the update
%           equations.
%       config.G_sparsity: [non-negative scalar] (default: 0)
%           sparsity level for the time-varying convex combination matrix.
%       config.H_sparsity: [non-negative scalar] (default: 0)
%           sparsity level for the encoding matrix.
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
%   W: [3D tensor]
%       m-by-num_basis_elems-by-context_len basis tensor. W = S*G.
%   H: [non-negative matrix]
%       num_basis_elems-by-n non-negative encoding matrix.
%   S: [matrix]
%       m-by-p matrix of p points belonging to the convex hull of V.
%   G: [non-negative 3D tensor]
%       p-by-num_basis_elems-by-context_len tensor of time-varying convex
%       combinations of the columns of S.
%   cost: [vector]
%       value of the cost function after each iteration.
%
% References:
%   [1] C. Vaz, A. Toutios, and S. Narayanan, "Convex Hull Convolutive
%       Non-negative Matrix Factorization for Uncovering Temporal Patterns
%       in Multivariate Time-Series Data," in Interspeech, San Francisco,
%       CA, 2016.
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
    elseif n <= 2  % If only have at most 2 points, then let convexhull be V
        config.S_init = V;
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
num_points = size(S, 2);

% Check if given initial basis (W \approx S * G)
if ~isfield(config, 'W_init') || isempty(config.W_init)
    given_W_init = false;
else
    given_W_init = true;
    W_init = config.W_init;
end

% Update switch for the basis
if ~isfield(config, 'W_fixed')
    config.W_fixed = false;
elseif config.W_fixed
    config.G_fixed = true;  % make G fixed if W is fixed because W \approx S * G
end

% Initialize convex combination tensor
if given_W_init
    % Find best G_init such that W_init \approx S * G_init
    config.G_init = rand(num_points, num_basis_elems, context_len);
%     for t = 1 : context_len
%         cvx_begin quiet
%             variable G_t(num_points, num_basis_elems)
%             minimize(norm(W_init(:, :, t) - S * G_t, 'fro'))
%             subject to
%                 G_t >= eps
%                 sum(G_t, 1) == 1
%         cvx_end
%         config.G_init(:, :, t) = G_t;
%     end
    
    for t = 1 : context_len
        config.G_init(:, :, t) = config.G_init(:, :, t) * diag(1 ./ sum(config.G_init(:, :, t), 1));
        S_W_pos = 0.5 * (abs(S' * W_init(:, :, t)) + (S' * W_init(:, :, t)));
        S_W_neg = 0.5 * (abs(S' * W_init(:, :, t)) - (S' * W_init(:, :, t)));
        S_S_pos = 0.5 * (abs(S' * S) + (S' * S));
        S_S_neg = 0.5 * (abs(S' * S) - (S' * S));
        prev_cost = inf;
        for inner_iter = 1 : 100
            config.G_init(:, :, t) = config.G_init(:, :, t) .* ((S_W_pos + S_S_neg*config.G_init(:, :, t)) ./ (S_W_neg + S_S_pos*config.G_init(:, :, t)));
            config.G_init(:, :, t) = config.G_init(:, :, t) * diag(1 ./ sum(config.G_init(:, :, t), 1));
            curr_cost = 0.5 * norm(W_init(:, :, t) - S * config.G_init(:, :, t), 'fro')^2;
            if curr_cost <= prev_cost && prev_cost - curr_cost <= 1e-5
                break;
            end
            prev_cost = curr_cost;
        end
    end
    
elseif ~isfield(config, 'G_init') || isempty(config.G_init)
    config.G_init = rand(num_points, num_basis_elems, context_len);
%     for t = 2 : context_len
%         config.G_init(:, :, t) = abs(config.G_init(:, :, t-1) + 0.1 * (2*rand(num_points, num_basis_elems) - 1));
%     end
    norms = zeros(num_basis_elems, context_len);
    for t = 1 : context_len
        norms(:, t) = sum(config.G_init(:, :, t), 1)';
        config.G_init(:, :, t) = config.G_init(:, :, t) * diag(1 ./ sum(config.G_init(:, :, t)));
    end
end
G = config.G_init;

% Update switch for convex combination tensor
if ~isfield(config, 'G_fixed')
    config.G_fixed = false;
end

% Sparsity level for convex combination tensor
if ~isfield(config, 'G_sparsity') || isempty(config.G_sparsity)
    config.G_sparsity = 0;
% elseif config.G_sparsity > 0  % Hoyer's sparsity constraint
%     L2s = 1 / (sqrt(m) - (sqrt(m) - 1) * config.G_sparsity);
%     for t = 1 : context_len
%         for k = 1 : num_basis_elems
%             G(:, k, t) = projfunc(G(:, k, t), 1, L2s, 1);
%         end
%     end
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
%     L1s = sqrt(n) - (sqrt(n) - 1) * config.H_sparsity;
%     for k = 1 : num_basis_elems
%         H_norm = norm(H(k, :));
%         H(k, :) = H(k, :) / H_norm;
%         H(k, :) = (projfunc(H(k, :)', L1s, 1, 1))';
%         H(k, :) = H(k, :) * H_norm;
%     end
end

if ~isfield(config, 'maxiter') || config.maxiter <= 0
    config.maxiter = 100;
end

% Maximum tolerance in cost function change per iteration
if ~isfield(config, 'tolerance') || config.tolerance <= 0
    config.tolerance = 1e-3;
end

G0 = G;

% if given_W_init
%     cost = zeros(config.maxiter+1, 1);
%     V_hat = ReconstructFromDecomposition(W, H);
%     cost(1) = 0.5 * sum(sum((V - V_hat).^2)) + config.H_sparsity * sum(sum(H));
%     
%     V_pos = 0.5 * (abs(V) + V);
%     V_neg = 0.5 * (abs(V) - V);
%     
%     for iter = 1 : config.maxiter
% %         if ~config.W_fixed
% %             W_pos = 0.5 * (abs(W) + W);
% %             W_neg = 0.5 * (abs(W) - W);
% %             V_hat_pos = ReconstructFromDecomposition(W_pos, H);
% %             V_hat_neg = ReconstructFromDecomposition(W_neg, H);
% %             for t = 1 : context_len
% %                 H_shifted = [zeros(num_basis_elems, t-1) H(:, 1:n-t+1)];
% %                 
% %                 W(:, :, t) = W(:, :, t) .* (((V_pos + V_hat_neg) * H_shifted') ./ max((V_neg + V_hat_pos) * H_shifted', eps));
% %             end
% %         end
%         
% %         if ~config.H_fixed
%             grad_neg = zeros(num_basis_elems, n);
%             grad_pos = zeros(num_basis_elems, n);
% 
%             for t1 = 1 : context_len
%                 V_shifted = [V(:, t1:n) zeros(m, t1-1)];
%                 W_V_pos = 0.5 * (abs(W(:, :, t1)' * V_shifted) + (W(:, :, t1)' * V_shifted));
%                 W_V_neg = 0.5 * (abs(W(:, :, t1)' * V_shifted) - (W(:, :, t1)' * V_shifted));
%                 W_V_hat_neg = zeros(num_basis_elems, n);
%                 W_V_hat_pos = zeros(num_basis_elems, n);
% 
%                 for t2 = 1 : context_len
%                     if t2 >= t1
%                         H_shifted = [zeros(num_basis_elems, t2-t1) H(:, 1:n-(t2-t1))];
%                     else
%                         H_shifted = [H(:, (num_basis_elems-t2)+1:n) zeros(k, t1-t2)];
%                     end
%                     W_W_pos = 0.5 * (abs(W(:, :, t1)' * W(:, :, t2)) + (W(:, :, t1)' * W(:, :, t2)));
%                     W_W_neg = 0.5 * (abs(W(:, :, t1)' * W(:, :, t2)) - (W(:, :, t1)' * W(:, :, t2)));
%                     W_V_hat_neg = W_V_hat_neg + W_W_neg * H_shifted;
%                     W_V_hat_pos = W_V_hat_pos + W_W_pos * H_shifted;
%                 end
%                 grad_neg = grad_neg + W_V_pos + W_V_hat_neg;
%                 grad_pos = grad_pos + W_V_neg + W_V_hat_pos;
%             end
%             H = H .* (grad_neg ./ max(grad_pos + config.H_sparsity, eps));
% %         end
% 
%         V_hat = ReconstructFromDecomposition(W, H);
%         cost(iter+1) = 0.5 * sum(sum((V - V_hat).^2)) + config.H_sparsity * sum(sum(H));
%     end
% else
    S_V_pos = 0.5 * (abs(S' * V) + (S' * V));
    S_V_neg = 0.5 * (abs(S' * V) - (S' * V));
    S_S_pos = 0.5 * (abs(S' * S) + (S' * S));
    S_S_neg = 0.5 * (abs(S' * S) - (S' * S));
    identity_mat = speye(n);
    W = zeros(m, num_basis_elems, context_len);
    for t = 1 : context_len
        W(:, :, t) = S * G(:, :, t);
    end

    cost = zeros(config.maxiter+1, 1);
    V_hat = ReconstructFromDecomposition(W, H);
    cost(1) = 0.5 * sum(sum((V - V_hat).^2)) + config.H_sparsity * sum(sum(H));

    stepsizeG = ones(context_len, 1);
    stepsizeH = 1;

    for iter = 1 : config.maxiter
        F = ReconstructFromDecomposition(G0, H);

        % Update convex combination tensor
        if ~config.G_fixed
            norms = zeros(num_basis_elems, context_len);
            for t = 1 : context_len
                H_shifted = [zeros(num_basis_elems, t-1) H(:, 1:n-t+1)];

                % Hoyer's sparsity constraint
    %             if config.G_sparsity > 0
    %                 % Gradient for H
    %                 dG = (S_V_neg + S_S_pos * F) * H_shifted' - (S_V_pos + S_S_neg * F) * H_shifted';
    %                 W_current = W;
    %                 V_hat = ReconstructFromDecomposition(W_current, H);
    %                 begobj = 0.5 * sum(sum((V - V_hat).^2));
    % 
    %                 % Make sure we decrease the objective!
    %                 while 1
    %                     % Take step in direction of negative gradient, and project
    %                     Gnew = G0(:, :, t) - stepsizeG(t) * dG;
    %                     for k = 1 : num_basis_elems
    %                         Gnew(:, k) = projfunc(Gnew(:, k), 1, L2s, 1);
    %                     end
    % 
    %                     W_current(:, :, t) = S * Gnew;
    % 
    %                     % Calculate new objective
    %                     V_hat = ReconstructFromDecomposition(W_current, H);
    %                     newobj = 0.5 * sum(sum((V - V_hat).^2));
    % 
    %                     % If the objective decreased, we can continue...
    %                     if newobj <= begobj
    %                         break;
    %                     end
    % 
    %                     % ...else decrease stepsize and try again
    %                     stepsizeG(t) = stepsizeG(t) / 2;
    %                     if stepsizeG(t) < 1e-200
    %                         fprintf('Algorithm converged.\n');
    %                         cost = cost(1 : iter);  % trim
    %                         return; 
    %                     end
    %                 end
    % 
    %                 % Slightly increase the stepsize
    %                 stepsizeG(t) = 1.2 * stepsizeG(t);
    %                 G(:, :, t) = Gnew;
    %             else
                    G(:, :, t) = G0(:, :, t) .* (((S_V_pos + S_S_neg * F) * H_shifted') ./ max((S_V_neg + S_S_pos * F) * H_shifted' + config.G_sparsity, eps));
                    norms(:, t) = sum(G(:, :, t), 1)';
                    G(:, :, t) = G(:, :, t) * diag(1 ./ sum(G(:, :, t), 1));
    %             end
                F = max(F + (G(:, :, t) - G0(:, :, t)) * H_shifted, 0);
                W(:, :, t) = S * G(:, :, t);
            end
    %         H = context_len * diag(1 ./ sum((1 ./ norms), 2)) * H;
        end

        % Update encoding matrix
        if ~config.H_fixed
            F = ReconstructFromDecomposition(G, H);
            negative_grad = zeros(num_basis_elems, n);
            positive_grad = zeros(num_basis_elems, n);
            for t = 1 : context_len
                identity_shifted = [identity_mat(:, t:n) zeros(n, t-1)];
                F_shifted = [F(:, t:n) zeros(num_points, t-1)];
                negative_grad = negative_grad + G(:, :, t)' * (S_V_pos * identity_shifted + S_S_neg * F_shifted);
                positive_grad = positive_grad + G(:, :, t)' * (S_V_neg * identity_shifted + S_S_pos * F_shifted);
            end
            % Hoyer's sparsity constraint
    %         if config.H_sparsity > 0
    %             % Gradient for H
    %             dH = positive_grad - negative_grad;
    %             begobj = cost(iter);
    % 
    %             % Make sure we decrease the objective!
    %             while 1
    %                 % Take step in direction of negative gradient, and project
    %                 Hnew = H - stepsizeH * dH;
    %                 for k = 1 : num_basis_elems
    %                     H_norm = norm(Hnew(k, :));
    %                     Hnew(k, :) = Hnew(k, :) / H_norm;
    %                     Hnew(k, :) = (projfunc(Hnew(k, :)', L1s, 1, 1))';
    %                     Hnew(k, :) = Hnew(k, :) * H_norm;
    %                 end
    % 
    %                 % Calculate new objective
    %                 V_hat = ReconstructFromDecomposition(W, Hnew);
    %                 newobj = 0.5 * sum(sum((V - V_hat).^2));
    % 
    %                 % If the objective decreased, we can continue...
    %                 if newobj <= begobj
    %                     break;
    %                 end
    % 
    %                 % ...else decrease stepsize and try again
    %                 stepsizeH = stepsizeH / 2;
    %                 if stepsizeH < 1e-200
    %                     fprintf('Algorithm converged.\n');
    %                     cost = cost(1 : iter);  % trim
    %                     return; 
    %                 end
    %             end
    % 
    %             % Slightly increase the stepsize
    %             stepsizeH = 1.2 * stepsizeH;
    %             H = Hnew;
    %         else
                H = H .* (negative_grad ./ max(positive_grad + config.H_sparsity, eps));
    %         end
        end

        % Calculate cost for this iteration
        V_hat = ReconstructFromDecomposition(W, H);
        cost(iter+1) = 0.5 * sum(sum((V - V_hat).^2)) + config.H_sparsity * sum(sum(H));

        % Stop iterations if change in cost function less than the tolerance
        if iter > 1 && cost(iter+1) < cost(iter) && cost(iter) - cost(iter+1) < config.tolerance
            cost = cost(1 : iter+1);  % trim vector
            break;
        end

        G0 = G;
    end
% end

end  % function
