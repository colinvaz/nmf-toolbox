function V_hat = ReconstructFromDecomposition(W, H)
% ReconstructFromDecomposition Reconstruct the input signal from the
% NMF/CNMF basis and encoding matrix.
%
% Inputs:
%   W: [non-negative matrix] or [non-negative 3D tensor]
%       [non-negative matrix]: m-by-num_basis_elems NMF basis.
%       [non-negative 3D tensor]: m-by-num_basis_elems-by-num_frames CNMF
%       basis.
%   H: [non-negative matrix]
%       num_dict_elems-by-n encoding matrix.
%
% Output:
%   V_hat: [non-negative matrix]
%       m-by-n reconstructed input matrix.
%
% NMF Toolbox
% Colin Vaz - cvaz@usc.edu
% Signal Analysis and Interpretation Lab (SAIL) - http://sail.usc.edu
% University of Southern California
% 2015

if iscell(W)
    W = cell2mat(W);
end
if iscell(H)
    H = cell2mat(H);
end

if ismatrix(W)  % nmf
    V_hat = W * H;
elseif ndims(W) == 3  % cnmf
    [m, num_basis_elems, num_frames] = size(W);
    n = size(H, 2);
    V_hat = zeros(m, n);
    for t = 1 : num_frames
        V_hat = V_hat + W(:, :, t) * [zeros(num_basis_elems, t-1) H(:, 1:n-t+1)];
    end
end

end  % function
