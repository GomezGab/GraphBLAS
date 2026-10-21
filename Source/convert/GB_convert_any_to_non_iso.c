//------------------------------------------------------------------------------
// GB_convert_any_to_non_iso: convert a matrix from iso to non-iso
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#define GB_FREE_ALL ;

GrB_Info GB_convert_any_to_non_iso // convert iso matrix to non-iso
(
    GrB_Matrix A,           // input/output matrix
    bool initialize         // if true, copy the iso value to all of A->x;
                            // otherise, just copy it back to A->x [0].
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    ASSERT_MATRIX_OK (A, "A to convert to non-iso", GB0) ;
    if (!A->iso)
    { 
        // nothing to do
        return (GrB_SUCCESS) ;
    }

    //--------------------------------------------------------------------------
    // get the iso entry of A
    //--------------------------------------------------------------------------

    size_t asize = A->type->size ;
    GB_void scalar [GB_VLA(asize)] ;
    memcpy (scalar, A->x, asize) ;

    //--------------------------------------------------------------------------
    // ensure A->x is large enough, and not shallow
    //--------------------------------------------------------------------------

    int64_t anz = GB_nnz_held (A) ;
    anz = GB_IMAX (anz, 1) ;
    int64_t Ax_memsize_required = anz * asize ;

    if (GB_array_memsize (GB_CSLOT (A, x)) < Ax_memsize_required
        || A->x_shallow || GB_array_is_shared (GB_CSLOT (A, x)))
    {
        // release the old space (freed unless it is shallow or shared)
        GB_array_release (GB_SLOT (A, x)) ;
        // allocate the new space
        A->x = GB_array_malloc (GB_SLOT (A, x), anz, asize, A->data_arena) ;
        if (A->x == NULL)
        { 
            // out of memory
            GB_phybix_free (A) ;
            return (GrB_OUT_OF_MEMORY) ;
        }
    }

    //--------------------------------------------------------------------------
    // copy the first entry into all of A->x, or to just A->x [0]
    //--------------------------------------------------------------------------

    if (initialize)
    { 
        // Ax [0:anz-1] = scalar
        GB_OK (GB_iso_expand (A->x, anz, scalar, A->type)) ;
    }
    else
    { 
        // Ax [0] = scalar
        memcpy (A->x, scalar, asize) ;
    }

    //--------------------------------------------------------------------------
    // create A->Pending->x if any pending tuples present
    //--------------------------------------------------------------------------

    if (A->Pending != NULL)
    { 
        // allocate A->Pending->x, which is currently NULL since Pending->x is
        // NULL if and only if A is iso, when Pending tuples are present.
        // The A->Pending->op is NULL and remains so.
        ASSERT (A->Pending->x == NULL) ;
        ASSERT (GB_memsize (A->Pending->x_mem) == 0) ;
        ASSERT (A->Pending->op == NULL) ;
        A->Pending->type = A->type ;
        A->Pending->size = A->type->size ;
        A->Pending->x = GB_MALLOC_MEMORY (A->Pending->nmax, A->Pending->size,
            &(A->Pending->x_mem)) ;
        if (A->Pending->x == NULL)
        { 
            return (GrB_OUT_OF_MEMORY) ;
        }
        // Pending_x [0:(A->Pending->n)-1] = scalar
        GB_OK (GB_iso_expand (A->Pending->x, A->Pending->n, scalar, A->type)) ;
    }

    //--------------------------------------------------------------------------
    // finalize the matrix and return result
    //--------------------------------------------------------------------------

    A->iso = false ;
    ASSERT_MATRIX_OK (A, "A converted to non-iso", GB0) ;
    return (GrB_SUCCESS) ;
}

