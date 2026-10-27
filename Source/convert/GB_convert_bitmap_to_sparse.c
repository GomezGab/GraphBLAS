//------------------------------------------------------------------------------
// GB_convert_bitmap_to_sparse: convert a matrix from bitmap to sparse
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

#define GB_FREE_ALL                     \
{                                       \
    GB_array_release (GB_VSLOT (Cp)) ;  \
    GB_array_release (GB_VSLOT (Ci)) ;  \
    GB_array_release (GB_VSLOT (Cx)) ;  \
}

GrB_Info GB_convert_bitmap_to_sparse    // convert matrix from bitmap to sparse
(
    GrB_Matrix A,               // matrix to convert from bitmap to sparse
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    ASSERT_MATRIX_OK (A, "A converting bitmap to sparse", GB0) ;
    ASSERT (!GB_IS_FULL (A)) ;
    ASSERT (GB_IS_BITMAP (A)) ;
    ASSERT (!GB_IS_SPARSE (A)) ;
    ASSERT (!GB_IS_HYPERSPARSE (A)) ;
    ASSERT (!GB_PENDING (A)) ;      // bitmap never has pending tuples
    ASSERT (!GB_JUMBLED (A)) ;      // bitmap is never jumbled
    ASSERT (!GB_ZOMBIES (A)) ;      // bitmap never has zomies

    int data_arena = A->data_arena ;

    //--------------------------------------------------------------------------
    // allocate Cp, Ci, and Cx
    //--------------------------------------------------------------------------

    const int64_t anvals = A->nvals ;
    GB_BURBLE_N (anvals, "(bitmap to sparse) ") ;
    const int64_t anzmax = GB_IMAX (anvals, 1) ;
    int64_t cnvec_nonempty ;
    const int64_t avdim = A->vdim ;
    const size_t asize = A->type->size ;
    GB_VSLOT_DECLARE (void *, Cp) ;
    GB_VSLOT_DECLARE (void *, Ci) ;
    GB_VSLOT_DECLARE (void *, Cx) ;

    bool Cp_is_32, Cj_is_32, Ci_is_32 ;
    GB_determine_pji_is_32 (&Cp_is_32, &Cj_is_32, &Ci_is_32,
        GxB_AUTO_SPARSITY, anzmax, A->vlen, avdim, Werk) ;

    size_t psize = Cp_is_32 ? sizeof (uint32_t) : sizeof (uint64_t) ;
    size_t isize = Ci_is_32 ? sizeof (uint32_t) : sizeof (uint64_t) ;

    Cp = GB_array_malloc (GB_VSLOT (Cp), avdim+1, psize, data_arena) ;
    Ci = GB_array_malloc (GB_VSLOT (Ci), anzmax,  isize, data_arena) ;
    if (Cp == NULL || Ci == NULL)
    { 
        // out of memory
        GB_FREE_ALL ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    const bool A_iso = A->iso ;
    if (!A_iso)
    { 
        // A is not iso.  Allocate new space for Cx, which is filled by
        // GB_convert_b2s.
        Cx = GB_array_malloc (GB_VSLOT (Cx), anzmax, asize, data_arena) ;
        if (Cx == NULL)
        { 
            // out of memory
            GB_FREE_ALL ;
            return (GrB_OUT_OF_MEMORY) ;
        }
    }

    //--------------------------------------------------------------------------
    // convert to sparse format (Cp, Ci, and Cx)
    //--------------------------------------------------------------------------

    // Cx and A->x always have the same type.
    // The values are not converted if A is iso (Cx is NULL).
    GB_OK (GB_convert_b2s (Cp, Ci, NULL, Cx, &cnvec_nonempty,
        Cp_is_32, false, Ci_is_32, A->type, A, data_arena, Werk)) ;

    //--------------------------------------------------------------------------
    // free prior content of A and transplant the new content
    //--------------------------------------------------------------------------

    if (A_iso)
    {
        // A is iso.  Remove A->x from the matrix so it is not freed by
        // GB_phybix_free; it is transplanted back again just below.
        GB_array_move (GB_VSLOT (Cx), GB_SLOT (A, x)) ;
    }

    GB_phybix_free (A) ;        // clears A->nvals
    GB_array_move (GB_SLOT (A, p), GB_VSLOT (Cp)) ;
    GB_array_move (GB_SLOT (A, i), GB_VSLOT (Ci)) ;
    GB_array_move (GB_SLOT (A, x), GB_VSLOT (Cx)) ;
    A->p_is_32 = Cp_is_32 ;
    A->j_is_32 = Cj_is_32 ;
    A->i_is_32 = Ci_is_32 ;
    A->iso = A_iso ;
    A->nvals = anvals ;
    A->plen = avdim ;
    A->nvec = avdim ;
    GB_nvec_nonempty_set (A, cnvec_nonempty) ;
    A->magic = GB_MAGIC ;

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    ASSERT_MATRIX_OK (A, "A converted from to bitmap to sparse", GB0) ;
    ASSERT (GB_IS_SPARSE (A)) ;
    ASSERT (!GB_ZOMBIES (A)) ;
    ASSERT (!GB_JUMBLED (A)) ;
    ASSERT (!GB_PENDING (A)) ;
    return (GrB_SUCCESS) ;
}

