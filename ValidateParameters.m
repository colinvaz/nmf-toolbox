function [config_out, is_W_cell, is_H_cell] = ValidateParameters(algorithm, config_in, V, num_basis_elems, context_len)
% ValidateParameters Validate NMF parameters.
%
% NMF Toolbox
% Colin Vaz - cvaz@usc.edu
% Signal Analysis and Interpretation Lab (SAIL) - http://sail.usc.edu
% University of Southern California
% 2015

config_out = config_in;

[m, n] = size(V);

num_sources = length(num_basis_elems);

if ~isfield(config_out, 'divergence')
    config_out.divergence = 'euclidean';
end

if ~isfield(config_out, 'alpha')
    config_out.alpha = 1;
elseif ~strcmp(config_out.divergence, 'ab_divergence') && ~strcmp(config_out.divergence, 'ab')
    config_out.alpha = 1;
end

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
    switch algorithm
        case {'nmf', 'cnmf', 'cmf', 'ccmf', 'cmfwisa', 'ccmfwisa'}
            for i = 1 : num_sources
                config_out.H_init{i} = max(rand(num_basis_elems{i}, n), eps);
            end
        case {'seminmf', 'convexnmf', 'convexhullnmf', 'convexhullcnmf', 'convexhullccmf'}
            % TODO: check if this works for multiple dictionaries/activation matrices
            for i = 1 : num_sources
                cluster_idx = kmeans(V.', num_basis_elems{i});
                config_out.H_init{i} = zeros(num_basis_elems{i}, n);
                for j = 1 : n
                    config_out.H_init(cluster_idx(j), j) = 1;
                end
                config_out.H_init{i} = config_out.H_init{i} + 0.2;
            end
        otherwise
            error(['Algorithm ', algorithm, ' is not defined in this toolbox.']);
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
    switch algorithm
        case {'nmf', 'cmfwisa'}
            for i = 1 : num_sources
                config_out.W_init{i} = max(rand(m, num_basis_elems{i}), eps);
                config_out.W_init{i} = config_out.W_init{i} * diag(1 ./ sqrt(sum(config_out.W_init{i}.^2, 1)));
            end
        case {'cnmf', 'ccmfwisa'}
            for i = 1 : num_sources
                config_out.W_init{i} = rand(m, num_basis_elems{i}, context_len);
                for t = 1 : context_len
                    config_out.W_init{i}(:, :, t) = config_out.W_init{i}(:, :, t) * diag(1 ./ sqrt(sum(config_out.W_init{i}(:, :, t).^2, 1)));
                end
            end
        case 'cmf'
            for i = 1 : num_sources
                config_out.W_init{i} = (2 * rand(m, num_basis_elems{i}) - 1) + 1j*(2 * rand(m, num_basis_elems{i}) - 1);
                config_out.W_init{i} = config_out.W_init{i} * diag(1 ./ sqrt(real(diag(config_out.W_init{i}' * config_out.W_init{i}))));
            end
        case 'ccmf'
            for i = 1 : num_sources
                config_out.W_init{i} = (2 * rand(m, num_basis_elems{i}, context_len) - 1) + 1j*(2 * rand(m, num_basis_elems{i}, context_len) - 1);
                for t = 1 : context_len
                    config_out.W_init{i}(:, :, t) = config_out.W_init{i}(:, :, t) * diag(1 ./ sqrt(real(diag(config_out.W_init{i}(:, :, t)' * config_out.W_init{i}(:, :, t)))));
                end
            end
        case 'seminmf'
            for i = 1 : num_sources
                config_out.W_init{i} = 2*rand(m, num_basis_elems{i}) - 1;
            end
        case {'convexnmf', 'convexhullnmf'}
            for i = 1 : num_sources
                H = config_out.H_init{i} - 0.2;
                config_out.W_init{i} = config_out.H_init{i}' * diag(1 ./ sum(H, 2));
            end
        case {'convexhullcnmf', 'convexhullccmf'}
            for i = 1 : num_sources
                config_out.W_init{i} = rand(m, num_basis_elems{i}, context_len);
                for t = 1 : context_len
                    config_out.W_init{i}(:, :, t) = config_out.W_init{i}(:, :, t) * diag(1 ./ sum(config_out.W_init{i}(:, :, t))); %V * config.H_init' * diag(1 ./ sum(H, 2));
                end
            end
        otherwise
            error(['Algorithm ', algorithm, ' is not defined in this toolbox.']);
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
    for i = 1 : num_sources
        config_out.W_sparsity{i} = 0;
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
    for i = 1 : num_sources
        config_out.W_sparsity{i} = temp;
    end
    clear temp;
else  % make sure all given sparsity levels are non-negative
    for i = 1 : num_sources
        config_out.W_sparsity{i} = max(config_out.W_sparsity{i}, 0);
    end
end

% Sparsity levels for encoding matrices
if ~isfield(config_out, 'H_sparsity') || isempty(config_out.H_sparsity)  % not given a sparsity level. Fill this in.
    config_out.H_sparsity = cell(num_sources, 1);
    for i = 1 : num_sources
        config_out.H_sparsity{i} = 0;
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
    for i = 1 : num_sources
        config_out.H_sparsity{i} = temp;
    end
    clear temp;
else  % make sure all given sparsity levels are non-negative
    for i = 1 : num_sources
        config_out.H_sparsity{i} = max(config_out.H_sparsity{i}, 0);
    end
end

% Update switches for basis matrices
if ~isfield(config_out, 'W_fixed') || isempty(config_out.W_fixed)  % not given an update switch. Fill this in.
    config_out.W_fixed = cell(num_sources, 1);
    for i = 1 : num_sources
        config_out.W_fixed{i} = false;
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
    for i = 1 : num_sources
        config_out.W_fixed{i} = temp;
    end
    clear temp;
end

% Update switches for encoding matrices
if ~isfield(config_out, 'H_fixed') || isempty(config_out.H_fixed)  % not given an update switch. Fill this in.
    config_out.H_fixed = cell(num_sources, 1);
    for i = 1 : num_sources
        config_out.H_fixed{i} = false;
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
    for i = 1 : num_sources
        config_out.H_fixed{i} = temp;
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
