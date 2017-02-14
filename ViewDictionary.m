function ViewDictionary(W, config)
% ViewDictioary Plot the basis from NMF or CNMF.
%
% For NMF, the basis is displayed with each basis element along the x axis
% and the data dimension along the y axis.
% For CNMF, the basis is first expanded along the temporal dimension, then
% each spectro-temporal basis element is plotted along the x axis and the
% data dimension along the y axis.
%
% Inputs:
%   W: [non-negative matrix] (from NMF) or [non-negative tensor] (from CNMF)
%       basis from NMF or CNMF.
%   config: [structure] (optional)
%       structure containing display options.
%       config.logscale: [boolean] (default: false)
%           indicate whether to plot the basis elements on the log-scale or
%           not. The log-scale usually helps with visualization.
%       config.flipud: [boolean] (default: false)
%           indicate whether to flip the y-axis when plotting.
%       config.threshold: [scalar] (default: -inf)
%           values below threshold are set to zero. Helps with removing
%           "noisy" parts and viewing the high-energy components.
%       config.sort: [boolean] (default: false)
%           indicate whether to sort the columns of the NMF basis in
%           ascending center of mass. Not sure how to do this for the CNMF
%           basis yet. 
%       config.spacing: [scalar] (default: 1)
%           number of columns of spacing between CNMF basis elements.
%
% NMF Toolbox
% Colin Vaz - cvaz@usc.edu
% Signal Analysis and Interpretation Lab (SAIL) - http://sail.usc.edu
% University of Southern California
% 2015

if nargin < 2
    config = struct;
end
if ~isfield(config, 'logscale')
    config.logscale = false;
end
if ~isfield(config, 'flipud')
    config.flipud = false;
end
if ~isfield(config, 'threshold')
    config.threshold = -inf;
end
if ~isfield(config, 'sort')
    config.sort = false;
end
if ~isfield(config, 'spacing') || config.spacing < 0
    config.spacing = 1;
end

% Determine if NMF or CNMF basis
if ismatrix(W)  % NMF
    if config.sort
        W = SortDictionary(W);
    end
    
    if config.logscale
        W_display = log10(max(W, config.threshold));
    else
        W_display = max(W, config.threshold);
    end
elseif ndims(W) == 3  % CNMF
    [m, K, T] = size(W);
    
    if config.logscale
        W_display = reshape(permute(cat(3, max(log10(W), config.threshold), -inf(m, K, config.spacing)), [1 3 2]), m, K*(T+config.spacing));
    else
        W_display = reshape(permute(cat(3, max(W, config.threshold), -inf(m, K, config.spacing)), [1 3 2]), m, K*(T+config.spacing));
    end
end

% Display basis
if config.flipud
    imagesc(W_display); axis xy; colorbar;
else
    imagesc(W_display); colorbar;
end
xlabel('Basis index');
if ndims(W) == 3
    set(gca, 'XTick', [round(4.5 * (T + config.spacing)) : 5 * (T + config.spacing) : size(W_display, 2)]);
    labels = cell(length([5 : 5 : K]), 1);
    for k = 1 : length(labels)
        labels{k} = num2str(5*k);
    end
    set(gca, 'XTickLabel', labels);
end

end  % function
