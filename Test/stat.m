function varargout = stat (varargin)
%STAT report status of statement coverage and malloc debugging
%
% [d nthreads] = stat
%
% This file shadows the builtin stat (file), which Octave 11 calls from dir
% and other library functions; a call with inputs is forwarded to the builtin.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin > 0)
    [varargout{1:max (nargout, 1)}] = builtin ('stat', varargin {:}) ;
    return ;
end

global GraphBLAS_debug GraphBLAS_grbcov

if (isempty (GraphBLAS_debug))
    GraphBLAS_debug = false ;
end

nthreads = nthreads_get ;

if (nargout == 0)
    fprintf ('malloc debug: %d  nthreads %d\n', GraphBLAS_debug, nthreads) ;
else
    varargout {1} = GraphBLAS_debug ;
    if (nargout >= 2)
        varargout {2} = nthreads ;
    end
end

if (~isempty (GraphBLAS_grbcov))
    covered = sum (GraphBLAS_grbcov > 0) ;
    n = length (GraphBLAS_grbcov) ;
    if (nargout == 0)
        fprintf ('test coverage: %d of %d (%0.4f%%)\n', ...
            covered, n, 100 * (covered / n)) ;
    end
end
