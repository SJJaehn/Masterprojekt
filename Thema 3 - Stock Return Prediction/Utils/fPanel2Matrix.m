function [mMatrix,vRowIDOut,vColIDout] = fPanel2Matrix(mPanel, vRowIDin, vColIDin, vRowIDOut, vColIDout)

% Default output IDs if not specified
if nargin < 4 || (isempty(vRowIDOut) && isempty(vColIDout))
    vRowIDOut = unique(vRowIDin);
    vColIDout = unique(vColIDin);
end

% Dimensions
iNumRows  = numel(vRowIDOut);
iNumCols  = numel(vColIDout);
iNumThird = size(mPanel, 2);

% Map row/column IDs to output indices
[~, vRowIdx] = ismember(vRowIDin, vRowIDOut);
[~, vColIdx] = ismember(vColIDin, vColIDout);

% Linear indices for assignment
vIndices = sub2ind([iNumRows, iNumCols], vRowIdx, vColIdx);

% Preallocate 3D output
if isa(mPanel,'datetime')
    mMatrix = NaT(iNumRows, iNumCols, iNumThird);
else
    mMatrix = NaN(iNumRows, iNumCols, iNumThird);
end

% Vectorized assignment:
% reshape mPanel to (N × 1 × T)
mPanel3D = reshape(mPanel, [], 1, iNumThird);

% Replicate indices for all T slices
idx3D = repmat(vIndices, 1, iNumThird);

% Convert to linear 3D indexing
idx3D = idx3D + (0:iNumThird-1) * (iNumRows * iNumCols);

% Fill output in one shot
mMatrix(idx3D) = mPanel3D;

end