# NMF Toolbox
Toolbox for performing Non-negative Matrix Factorization (NMF) and several variants. Currently, the code is in MATLAB, but there are plans to create a Python version and possibly a C/C++ version.

## Installation
Clone this repository or download a zip version. In MATLAB, add the NMF Toolbox directory to the path.

## Main functions
- `nmf.m`: Non-negative matrix factorization
- `nmfsc.m`: NMF with sparsity constraints
- `cnmf.m`: Convolutive NMF
- `cnmfsc.m`: Convolutive NMF with sparsity constraints
- `cmfwisa.m`: Complex NMF with intra-source additivity
- `lnmf.m`: Local NMF
- `convexnmf.m`: Convex NMF
- `seminmf.m`: Semi NMF
- `chnmf.m`: Convex Hull NMF
- `chcnmf.m`: Convex Hull Convolutive NMF

## Utility functions
- `ValidateParameters.m`: Check that input parameters are valid
- `ReconstructFromDecomposition.m`: Reconstruct input from a basis and encoding matrix
- `ViewDictionary.m`: Plot the basis for visualization
- `SortDictionary.m`: Order the basis vectors by increasing centroid
- `projfunc.m`: Projection function used by Hoyer's sparsity constraint

