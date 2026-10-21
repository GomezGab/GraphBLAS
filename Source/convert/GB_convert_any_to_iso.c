//------------------------------------------------------------------------------
// GB_convert_any_to_iso: convert a matrix from non-iso to iso
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// If the matrix is not iso, it is changed to iso with the scalar value
// provided or with A->x [0] if the scalar is NULL on input.  The A->x array is
// optionally compacted to be exactly large enough to hold the single scalar.

#include "GB.h"

GrB_Info GB_convert_any_to_iso // convert non-iso matrix to iso
(
    GrB_Matrix A,           // input/output matrix
    GB_void *scalar         // scalar value, of size A->type->size, or NULL
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT_MATRIX_OK (A, "A to convert to iso", GB0) ;

    //--------------------------------------------------------------------------
    // get the iso entry of A, or use the scalar provided on input
    //--------------------------------------------------------------------------

    size_t asize = A->type->size ;
    GB_void ascalar [GB_VLA(asize)] ;
    memset (ascalar, 0, asize) ;
    if (scalar == NULL && A->iso)
    { 
        memcpy (ascalar, A->x, asize) ;
    }

    //--------------------------------------------------------------------------
    // compact the matrix
    //--------------------------------------------------------------------------

    if (GB_array_memsize (GB_CSLOT (A, x)) != asize || A->x_shallow
        || GB_array_is_shared (GB_CSLOT (A, x)) || A->x == NULL)
    {

        //----------------------------------------------------------------------
        // release the old A->x and allocate the new A->x
        //----------------------------------------------------------------------

        // the old A->x is freed unless it is shallow or shared
        GB_array_release (GB_SLOT (A, x)) ;

        // allocate the new space
        A->x = GB_array_malloc (GB_SLOT (A, x), 1, asize, A->data_arena) ;
        if (A->x == NULL)
        { 
            // out of memory
            GB_phybix_free (A) ;
            return (GrB_OUT_OF_MEMORY) ;
        }
    }

    //--------------------------------------------------------------------------
    // copy the iso scalar into A->x [0]
    //--------------------------------------------------------------------------

    if (scalar == NULL)
    { 
        memcpy (A->x, ascalar, asize) ;
    }
    else
    { 
        memcpy (A->x, scalar, asize) ;
    }

    //--------------------------------------------------------------------------
    // finalize the matrix and return result
    //--------------------------------------------------------------------------

    A->iso = true ;
    ASSERT_MATRIX_OK (A, "A converted to iso", GB0) ;
    return (GrB_SUCCESS) ;
}

