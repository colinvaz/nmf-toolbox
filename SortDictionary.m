function [W_sorted, H_sorted] = SortDictionary(W, H)
% SortDictionary Sort NMF basis elements by increasing center of mass.
% Currently doesn't work for CNMF basis.
%
% Inputs:
%   W: [matrix]
%       basis to be sorted.
%   H: [matrix] (optional)
%       encoding matrix. The rows will be reordered according to the
%       reordering of the basis elements.
%
% Outputs:
%   W_sorted: [matrix]
%       sorted basis.
%   H_sorted: [matrix] (empty if H not provided as input)
%       encoding matrix with the rows reordered to match the reordering of
%       the basis.
%
% NMF Toolbox
% Colin Vaz - cvaz@usc.edu
% Signal Analysis and Interpretation Lab (SAIL) - http://sail.usc.edu
% University of Southern California
% 2015

if nargin < 2
	empty_H = true;
else
	empty_H = false;
end

num_basis_elems = size(W, 2);

W_sum = cumsum(W);
center_of_gravity = zeros(num_basis_elems, 1);
for j = 1 : num_basis_elems
	idx = find(W_sum(:, j) <= W_sum(end, j)/2, 1, 'last');
	if isempty(idx)
		center_of_gravity(j) = 1;
	else
    	center_of_gravity(j) = idx;
	end
end
[~, sorted] = sort(center_of_gravity);
W_sorted = W(:, sorted);
if ~empty_H
	H_sorted = H(sorted, :);
end

end  % function
