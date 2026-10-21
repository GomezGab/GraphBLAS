//------------------------------------------------------------------------------
// GB_export: export a matrix or vector (HISTORICAL)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// No conversion is done, except: the matrix A is moved to the data arena
// determined by the Context or global context if no Context is engaged (if not
// already there), A is convert to non-iso if requested, and all integers are
// converted to 64-bits.  The matrix is exported in its current sparsity
// structure and by-row/by-col format.

// If unpacking is true, the header arena of A remains unchanged.
// Otherwise, A is freed so it is no longer in any arena.

// All uses of this method are historical.

#include "import_export/GB_export.h"

#define GB_FREE_ALL                         \
{                                           \
    GB_array_release (GB_VSLOT (Ap_new)) ;  \
    GB_array_release (GB_VSLOT (Ah_new)) ;  \
}

GrB_Info GB_export      // export/unpack a matrix in any format
(
    bool unpacking,     // unpack if true, export and free if false.

    GrB_Matrix *A,      // handle of matrix to export and free, or unpack
    GrB_Type *type,     // type of matrix to export
    uint64_t *vlen,     // vector length
    uint64_t *vdim,     // vector dimension
    bool is_sparse_vector,      // true if A is a sparse GrB_Vector

    // the 5 arrays:
    uint64_t **Ap,      // pointers
    uint64_t *Ap_memsize,  // size of Ap in bytes

    uint64_t **Ah,      // vector indices
    uint64_t *Ah_memsize,  // size of Ah in bytes

    int8_t **Ab,        // bitmap
    uint64_t *Ab_memsize,  // size of Ab in bytes

    uint64_t **Ai,      // indices
    uint64_t *Ai_memsize,  // size of Ai in bytes

    void **Ax,          // values
    uint64_t *Ax_memsize,  // size of Ax in bytes

    // additional information for specific formats:
    uint64_t *nvals,    // # of entries for bitmap format.
    bool *jumbled,      // if true, sparse/hypersparse may be jumbled.
    uint64_t *nvec,     // size of Ah for hypersparse format.

    // information for all formats:
    int *sparsity,      // hypersparse, sparse, bitmap, or full
    bool *is_csc,       // if true then matrix is by-column, else by-row
    bool *iso,          // if true then A is iso and only one entry is returned
                        // in Ax, regardless of nvals(A).
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;

    int data_arena = GB_Context_data_arena ( ) ;
    int nthreads_max = GB_Context_nthreads_max ( ) ;

    int64_t *Ap_new = NULL ; uint64_t Ap_new_mem = 0 ; bool Ap_new_shallow = 0 ;
    int64_t *Ah_new = NULL ; uint64_t Ah_new_mem = 0 ; bool Ah_new_shallow = 0 ;
    uint64_t X_mem_out ;        // mem word of each unloaded array
    bool X_readonly ;           // always false: user matrices are never shallow
    ASSERT (A != NULL) ;
    GB_RETURN_IF_NULL (*A) ;

    //--------------------------------------------------------------------------
    // ensure the data_arena of A matches the Context data arena
    //--------------------------------------------------------------------------

    (*A)->data_arena = data_arena ;
    GB_OK (GB_wait_arenas (*A)) ;

    //--------------------------------------------------------------------------
    // ensure the matrix is all-64-bit
    //--------------------------------------------------------------------------

    GB_OK (GB_convert_int (*A, false, false, false, false)) ;

    //--------------------------------------------------------------------------
    // check more inputs
    //--------------------------------------------------------------------------

    GB_RETURN_IF_NULL_OR_INVALID (*A) ;
    ASSERT_MATRIX_OK (*A, "A to export", GB0) ;
    ASSERT (!GB_ZOMBIES (*A)) ;
    ASSERT (GB_JUMBLED_OK (*A)) ;
    ASSERT (!GB_PENDING (*A)) ;

    GB_RETURN_IF_NULL (type) ;
    GB_RETURN_IF_NULL (vlen) ;
    GB_RETURN_IF_NULL (vdim) ;
    GB_RETURN_IF_NULL (Ax) ;
    GB_RETURN_IF_NULL (Ax_memsize) ;

    int s = GB_sparsity (*A) ;

    switch (s)
    {
        case GxB_HYPERSPARSE : 
            GB_RETURN_IF_NULL (nvec) ;
            GB_RETURN_IF_NULL (Ah) ; GB_RETURN_IF_NULL (Ah_memsize) ;
            // fall through to the sparse case

        case GxB_SPARSE : 
            if (is_sparse_vector)
            { 
                GB_RETURN_IF_NULL (nvals) ;
            }
            else
            { 
                GB_RETURN_IF_NULL (Ap) ; GB_RETURN_IF_NULL (Ap_memsize) ;
            }
            GB_RETURN_IF_NULL (Ai) ; GB_RETURN_IF_NULL (Ai_memsize) ;
            break ;

        case GxB_BITMAP : 
            GB_RETURN_IF_NULL (nvals) ;
            GB_RETURN_IF_NULL (Ab) ; GB_RETURN_IF_NULL (Ab_memsize) ;
            // fall through to the full case

        case GxB_FULL : 
            break ;

        default: ;
    }

    //--------------------------------------------------------------------------
    // allocate new space for Ap and Ah if unpacking
    //--------------------------------------------------------------------------

    int64_t avdim = (*A)->vdim ;
    int64_t plen_new, nvec_new ;
    if (unpacking)
    {
        plen_new = (avdim == 0) ? 0 : 1 ;
        nvec_new = (avdim == 1) ? 1 : 0 ;
        Ap_new = GB_array_calloc (GB_VSLOT (Ap_new), plen_new+1,
            sizeof (int64_t), data_arena) ;
        if (avdim > 1)
        {
            // A is sparse if avdim <= 1, hypersparse if avdim > 1
            Ah_new = GB_array_calloc (GB_VSLOT (Ah_new), 1, sizeof (int64_t),
                data_arena) ;
        }
        if (Ap_new == NULL || (avdim > 1 && Ah_new == NULL))
        { 
            // out of memory
            GB_FREE_ALL ;
            return (GrB_OUT_OF_MEMORY) ;
        }
    }

    //--------------------------------------------------------------------------
    // ensure A is non-iso if requested, or export A as-is
    //--------------------------------------------------------------------------

    if (iso == NULL)
    {
        // ensure A is non-iso
        if ((*A)->iso)
        { 
            GBURBLE ("(iso to non-iso export) ") ;
        }
        GB_OK (GB_convert_any_to_non_iso (*A, true)) ;
        ASSERT (!((*A)->iso)) ;
    }
    else
    {
        // do not convert the matrix; export A as-is, either iso or non-iso
        (*iso) = (*A)->iso ;
        if (*iso)
        { 
            GBURBLE ("(iso export) ") ;
        }
    }

    //--------------------------------------------------------------------------
    // export the matrix
    //--------------------------------------------------------------------------

    (*type) = (*A)->type ;
    (*vlen) = (*A)->vlen ;
    (*vdim) = avdim ;

    // export A->x (copied first if it is shared with another matrix)
    GB_OK (GB_array_unload (GB_SLOT (*A, x), Ax, &X_mem_out, &X_readonly,
        nthreads_max, true, NULL)) ;
    ASSERT (!X_readonly) ;
    (*Ax_memsize) = GB_memsize (X_mem_out) ;

    switch (s)
    {
        case GxB_HYPERSPARSE : 
            (*nvec) = (*A)->nvec ;

            // export A->h
            GB_OK (GB_array_unload (GB_SLOT (*A, h), (void **) Ah, &X_mem_out,
                &X_readonly, nthreads_max, true, NULL)) ;
            ASSERT (!X_readonly) ;
            (*Ah_memsize) = GB_memsize (X_mem_out) ;
            // fall through to the sparse case

        case GxB_SPARSE : 
            if (jumbled != NULL)
            { 
                (*jumbled) = (*A)->jumbled ;
            }

            // export A->p, unless A is a sparse vector in CSC format
            if (is_sparse_vector)
            { 
                uint64_t *restrict Ap = (*A)->p ;       // OK; 64-bit only
                (*nvals) = Ap [1] ;
            }
            else
            { 
                GB_OK (GB_array_unload (GB_SLOT (*A, p), (void **) Ap,
                    &X_mem_out, &X_readonly, nthreads_max, true, NULL)) ;
                ASSERT (!X_readonly) ;
                (*Ap_memsize) = GB_memsize (X_mem_out) ;
            }

            // export A->i
            GB_OK (GB_array_unload (GB_SLOT (*A, i), (void **) Ai, &X_mem_out,
                &X_readonly, nthreads_max, true, NULL)) ;
            ASSERT (!X_readonly) ;
            (*Ai_memsize) = GB_memsize (X_mem_out) ;
            break ;

        case GxB_BITMAP : 
            (*nvals) = (*A)->nvals ;

            // export A->b
            GB_OK (GB_array_unload (GB_SLOT (*A, b), (void **) Ab, &X_mem_out,
                &X_readonly, nthreads_max, true, NULL)) ;
            ASSERT (!X_readonly) ;
            (*Ab_memsize) = GB_memsize (X_mem_out) ;

        case GxB_FULL : 

        default: ;
    }

    if (sparsity != NULL)
    { 
        (*sparsity) = s ;
    }
    if (is_csc != NULL)
    { 
        (*is_csc) = (*A)->is_csc ;
    }

    //--------------------------------------------------------------------------
    // free or clear the GrB_Matrix
    //--------------------------------------------------------------------------

    // both export and unpack free the hyper_hash, A->Y

    if (unpacking)
    { 
        // unpack: clear the matrix, leaving it hypersparse (or sparse if
        // it is a vector (vdim of 1) or has vdim of zero)
        GB_phybix_free (*A) ;
        (*A)->plen = plen_new ;
        (*A)->nvec = nvec_new ;
        GB_array_move (GB_SLOT (*A, p), GB_VSLOT (Ap_new)) ;
        GB_array_move (GB_SLOT (*A, h), GB_VSLOT (Ah_new)) ;
        (*A)->magic = GB_MAGIC ;
        (*A)->p_is_32 = false ;
        (*A)->j_is_32 = false ;
        (*A)->i_is_32 = false ;
        ASSERT_MATRIX_OK (*A, "A unpacked", GB0) ;
    }
    else
    { 
        // GxB_export: free the header of A, and A->p if A is a sparse
        // GrB_Vector.
        GB_Matrix_free (A) ;
        ASSERT ((*A) == NULL) ;
    }

    #pragma omp flush
    return (GrB_SUCCESS) ;
}

