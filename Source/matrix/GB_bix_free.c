//------------------------------------------------------------------------------
// GB_bix_free: free A->(b,i,x) pending tuples, zombies; A->p, A->h unchanged
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Since A->p and A->h are unchanged, the matrix is still valid (unless it was
// invalid on input).  nnz(A) would report zero, and so would GrB_Matrix_nvals.

// A->p_is_32, A->j_is_32, and A->i_is_32 are unchanged.

#include "pending/GB_Pending.h"

void GB_bix_free                // free A->b, A->i, and A->x of a matrix
(
    GrB_Matrix A                // matrix with content to free
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    if (A == NULL)
    { 
        // nothing to do
        return ;
    }

    //--------------------------------------------------------------------------
    // free all but A->p and A->h
    //--------------------------------------------------------------------------

    // zombies and pending tuples are about to be deleted
    ASSERT (GB_ZOMBIES_OK (A)) ;
    ASSERT (GB_PENDING_OK (A)) ;
    ASSERT (GB_JUMBLED_OK (A)) ;

    // release A->b, A->i, and A->x (freed only if not shallow or shared)
    GB_array_release (GB_SLOT (A, b)) ;
    GB_array_release (GB_SLOT (A, i)) ;
    // GB_nnz_max (A) will report zero if A->x is NULL:
    GB_array_release (GB_SLOT (A, x)) ;
    A->iso = false ;

    A->nvals = 0 ;

    // no zombies remain
    A->nzombies = 0 ;

    // an empty matrix is not jumbled
    A->jumbled = false ;

    // free the list of pending tuples
    GB_Pending_free (&(A->Pending)) ;
}

