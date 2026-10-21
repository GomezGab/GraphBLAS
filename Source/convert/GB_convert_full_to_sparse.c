//------------------------------------------------------------------------------
// GB_convert_full_to_sparse: convert a matrix from full to sparse
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

GrB_Info GB_convert_full_to_sparse      // convert matrix from full to sparse
(
    GrB_Matrix A,               // matrix to convert from full to sparse
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT_MATRIX_OK (A, "A converting full to sparse", GB0) ;
    ASSERT (GB_IS_FULL (A) || GB_nnz_max (A) == 0) ;
    ASSERT (!GB_IS_BITMAP (A)) ;
    ASSERT (!GB_IS_SPARSE (A)) ;
    ASSERT (!GB_IS_HYPERSPARSE (A)) ;
    ASSERT (!GB_ZOMBIES (A)) ;
    ASSERT (!GB_JUMBLED (A)) ;
    ASSERT (!GB_PENDING (A)) ;

    int data_arena = A->data_arena ;

    //--------------------------------------------------------------------------
    // allocate A->p and A->i
    //--------------------------------------------------------------------------

    int64_t avdim = A->vdim ;
    int64_t avlen = A->vlen ;
    int64_t anz = GB_nnz_full (A) ;
    GB_BURBLE_N (anz, "(full to sparse) ") ;

    bool Ap_is_32, Aj_is_32, Ai_is_32 ;
    GB_determine_pji_is_32 (&Ap_is_32, &Aj_is_32, &Ai_is_32,
        GxB_AUTO_SPARSITY, anz, avlen, avdim, Werk) ;

    void *Ap = NULL ; uint64_t Ap_mem = 0 ; bool Ap_shallow = false ;
    void *Ai = NULL ; uint64_t Ai_mem = 0 ; bool Ai_shallow = false ;

    size_t psize = (Ap_is_32) ? sizeof (uint32_t) : sizeof (uint64_t) ;
    size_t isize = (Ai_is_32) ? sizeof (uint32_t) : sizeof (uint64_t) ;

    Ap = GB_array_malloc (GB_VSLOT (Ap), avdim+1, psize, data_arena) ;
    Ai = GB_array_malloc (GB_VSLOT (Ai), anz, isize, data_arena) ;
    if (Ap == NULL || Ai == NULL)
    {
        // out of memory
        GB_array_release (GB_VSLOT (Ap)) ;
        GB_array_release (GB_VSLOT (Ai)) ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    // Ap and Ai are moved into A after they are filled, below
    A->p_is_32 = Ap_is_32 ;
    A->j_is_32 = Aj_is_32 ;
    A->i_is_32 = Ai_is_32 ;
    A->plen = avdim ;
    A->nvec = avdim ;
    GB_nvec_nonempty_set (A, (avlen == 0) ? 0 : avdim) ;
    A->nvals = anz ;

    //--------------------------------------------------------------------------
    // determine the number of threads to use
    //--------------------------------------------------------------------------

    int nthreads_max = GB_Context_nthreads_max ( ) ;
    double chunk = GB_Context_chunk ( ) ;
    int nthreads = GB_nthreads (anz, chunk, nthreads_max) ;

    //--------------------------------------------------------------------------
    // fill the A->p and A->i pattern
    //--------------------------------------------------------------------------

    GB_IDECL (Ap, , u) ; GB_IPTR (Ap, Ap_is_32) ;
    GB_IDECL (Ai, , u) ; GB_IPTR (Ai, Ai_is_32) ;

    int64_t k ;
    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (k = 0 ; k <= avdim ; k++)
    { 
        // Ap [k] = k * avlen ;
        GB_ISET (Ap, k, k * avlen) ;
    }

    int64_t p ;
    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (p = 0 ; p < anz ; p++)
    { 
        // Ai [p] = p % avlen ;
        GB_ISET (Ai, p, p % avlen) ;
    }

    //--------------------------------------------------------------------------
    // transplant the new A->p and A->i into A
    //--------------------------------------------------------------------------

    GB_array_move (GB_SLOT (A, p), GB_VSLOT (Ap)) ;
    GB_array_move (GB_SLOT (A, i), GB_VSLOT (Ai)) ;

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    ASSERT_MATRIX_OK (A, "A converted from full to sparse", GB0) ;
    ASSERT (GB_IS_SPARSE (A)) ;
    ASSERT (!GB_ZOMBIES (A)) ;
    ASSERT (!GB_JUMBLED (A)) ;
    ASSERT (!GB_PENDING (A)) ;
    return (GrB_SUCCESS) ;
}

