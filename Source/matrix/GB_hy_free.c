//------------------------------------------------------------------------------
// GB_hy_free: free A->h and A->Y
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

void GB_hy_free                 // free A-h and A->Y of a matrix
(
    GrB_Matrix A                // matrix with content to free
)
{ 

    if (A != NULL)
    {
        // release A->h (freed only if not shallow or shared)
        GB_array_release (GB_SLOT (A, h)) ;
        GB_hyper_hash_free (A) ;
    }
}

