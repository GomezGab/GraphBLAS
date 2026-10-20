//------------------------------------------------------------------------------
// GB_Matrix_diag: construct a diagonal matrix from a vector
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C = diag (V, k): C is n-by-n, sparse or hypersparse, with one entry per
// vector.  Its arrays are shared with V where they are identical, instead of
// copied: C->x is V->x when the types match, and V->i is C->i (for k >= 0)
// and/or C->h (for k <= 0), once k has been adjusted for the CSR/CSC format
// of C.  C->p is the ramp 0..vnz and is always allocated.  A shared array is
// copied later only if C or V is modified in place (see GB_array.h).

#define GB_FREE_WORKSPACE   \
{                           \
    GB_Matrix_free (&T) ;   \
}

#define GB_FREE_ALL         \
{                           \
    GB_FREE_WORKSPACE ;     \
    GB_phybix_free (C) ;    \
}

#include "diag/GB_diag.h"

GrB_Info GB_Matrix_diag     // build a diagonal matrix from a vector
(
    GrB_Matrix C,           // output matrix
    const GrB_Matrix V_in,  // input vector (as an n-by-1 matrix)
    int64_t k,
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    ASSERT_MATRIX_OK (C, "C input for GB_Matrix_diag", GB0) ;
    ASSERT_MATRIX_OK (V_in, "V input for GB_Matrix_diag", GB0) ;
    ASSERT (GB_VECTOR_OK (V_in)) ;       // V_in is a vector on input
    ASSERT (!GB_any_aliased (C, V_in)) ; // C and V_in cannot be aliased
    ASSERT (!GB_IS_HYPERSPARSE (V_in)) ; // vectors cannot be hypersparse

    GrB_Matrix T = NULL ;

    int header_arena = GB_arena (C->header_mem) ;
    int data_arena = C->data_arena ;

    GrB_Type ctype = C->type ;
    int64_t n = V_in->vlen + GB_IABS (k) ;     // C must be n-by-n

    ASSERT (GB_NROWS (C) == GB_NCOLS (C))
    ASSERT (GB_NROWS (C) == n)
    ASSERT (GB_Type_compatible (ctype, V_in->type)) ;

    //--------------------------------------------------------------------------
    // finish any pending work in V_in and clear the output matrix C
    //--------------------------------------------------------------------------

    GB_MATRIX_WAIT (V_in) ;
    GB_phybix_free (C) ;

    //--------------------------------------------------------------------------
    // ensure V is not bitmap
    //--------------------------------------------------------------------------

    GrB_Matrix V ;
    if (GB_IS_BITMAP (V_in))
    { 
        // make an exact deep copy of V_in, then convert to CSC.  C shares
        // the arrays of T below; T is freed when this method returns, and C
        // then becomes their sole owner.
        GB_OK (GB_dup (&T, V_in, header_arena, data_arena, Werk)) ;
        GB_OK (GB_convert_bitmap_to_sparse (T, Werk)) ;
        V = T ;
    }
    else
    { 
        // use V_in as-is
        V = V_in ;
    }

    //--------------------------------------------------------------------------
    // handle the CSR/CSC format of C and determine position of diagonal
    //--------------------------------------------------------------------------

    const bool csc = C->is_csc ;
    if (!csc)
    { 
        // The kth diagonal of a CSC matrix is the same as the (-k)th diagonal
        // of the CSR format, so if C is CSR, negate the value of k.  Then
        // treat C as if it were CSC in the rest of this method.
        k = -k ;
    }

    int64_t kpositive, knegative ;
    if (k >= 0)
    { 
        kpositive = k ;
        knegative = 0 ;
    }
    else
    { 
        kpositive = 0 ;
        knegative = -k ;
    }

    //--------------------------------------------------------------------------
    // determine the format of C
    //--------------------------------------------------------------------------

    // C is sparse if V is dense and k == 0, and hypersparse otherwise
    const int64_t vnz = GB_nnz (V) ;
    const bool V_is_full = GB_as_if_full (V) ;
    const int C_sparsity = (V_is_full && k == 0) ? GxB_SPARSE : GxB_HYPERSPARSE;
    const bool C_is_hyper = (C_sparsity == GxB_HYPERSPARSE) ;
    const bool C_iso = V->iso ;
    if (C_iso)
    { 
        GBURBLE ("(iso diag) ") ;
    }
    const float bitmap_switch = C->bitmap_switch ;
    const int sparsity_control = C->sparsity_control ;

    // determine the p_is_32, j_is_32, and i_is_32 settings for the new matrix
    bool Cp_is_32, Cj_is_32, Ci_is_32 ;
    GB_determine_pji_is_32 (&Cp_is_32, &Cj_is_32, &Ci_is_32,
        C_sparsity, vnz, n, n, Werk) ;

    //--------------------------------------------------------------------------
    // decide which arrays of V can be shared with C
    //--------------------------------------------------------------------------

    // An array of V is shared only if V owns it (a borrowed array has no
    // reference count and its owner may free it at any time) and it already
    // lives in the data arena of C.

    // C->x is V->x if the types match.  V->x is NULL only if V is empty.
    const bool arena_ok = (data_arena == V->data_arena) &&
        !GB_arenas_will_wait (V) ;
    const bool share_x = arena_ok && (ctype == V->type) && !V->x_shallow
        && (V->x != NULL) ;

    // V->i is the row index of each entry of V, which is the row index of
    // each entry of C when k >= 0, and the column index of each vector of C
    // when k <= 0.  V->i is NULL if V is full (it exists if V merely has all
    // entries present, in which case it is the ramp 0..vnz-1).  A sparse C
    // has no C->h.  C adopts the integer size of V->i for each shared array,
    // which must be valid for the dimension n of C (n can exceed vlen).
    bool share_i = false, share_h = false ;
    if (arena_ok && !V->i_shallow && (V->i != NULL))
    {
        share_i = (k >= 0) ;
        share_h = (k <= 0) && C_is_hyper ;
        bool Cj_shared = share_h ? V->i_is_32 : Cj_is_32 ;
        bool Ci_shared = share_i ? V->i_is_32 : Ci_is_32 ;
        if (GB_valid_pji_is_32 (Cp_is_32, Cj_shared, Ci_shared, vnz, n, n))
        { 
            Cj_is_32 = Cj_shared ;
            Ci_is_32 = Ci_shared ;
        }
        else
        { 
            // V->i is 32-bit but C is too large for 32-bit indices
            share_i = false ;
            share_h = false ;
        }
    }

    GBURBLE ("(diag shares:%s%s%s) ",
        share_h ? " h" : "", share_i ? " i" : "", share_x ? " x" : "") ;

    //--------------------------------------------------------------------------
    // allocate C as sparse or hypersparse with vnz entries and vnz vectors
    //--------------------------------------------------------------------------

    GB_OK (GB_new (&C, // existing header
        ctype, n, n, GB_ph_null, csc, C_sparsity, C->hyper_switch, vnz,
        Cp_is_32, Cj_is_32, Ci_is_32, header_arena, data_arena)) ;
    C->iso = C_iso ;
    C->sparsity_control = sparsity_control ;
    C->bitmap_switch = bitmap_switch ;

    size_t cpsize = Cp_is_32 ? sizeof (uint32_t) : sizeof (uint64_t) ;
    size_t cjsize = Cj_is_32 ? sizeof (uint32_t) : sizeof (uint64_t) ;
    size_t cisize = Ci_is_32 ? sizeof (uint32_t) : sizeof (uint64_t) ;
    const int64_t plen = C->plen ;
    const int64_t nzmax = GB_IMAX (vnz, 1) ;

    // C->p is always the ramp 0..vnz, and is always allocated
    C->p = GB_array_malloc (GB_SLOT (C, p), plen+1, cpsize, data_arena) ;
    if (C->p == NULL)
    { 
        // out of memory
        GB_FREE_ALL ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    // C->h is V->i (k <= 0), or allocated and computed below
    if (C_is_hyper)
    {
        if (share_h)
        { 
            GB_array_share (GB_SLOT (C, h), GB_CSLOT (V, i)) ;
        }
        else
        {
            C->h = GB_array_malloc (GB_SLOT (C, h), plen, cjsize, data_arena) ;
            if (C->h == NULL)
            { 
                // out of memory
                GB_FREE_ALL ;
                return (GrB_OUT_OF_MEMORY) ;
            }
        }
    }

    // C->i is V->i (k >= 0), or allocated and computed below
    if (share_i)
    { 
        GB_array_share (GB_SLOT (C, i), GB_CSLOT (V, i)) ;
    }
    else
    {
        C->i = GB_array_malloc (GB_SLOT (C, i), nzmax, cisize, data_arena) ;
        if (C->i == NULL)
        { 
            // out of memory
            GB_FREE_ALL ;
            return (GrB_OUT_OF_MEMORY) ;
        }
    }

    // C->x is V->x, or allocated and typecasted from V->x
    if (share_x)
    { 
        GB_array_share (GB_SLOT (C, x), GB_CSLOT (V, x)) ;
    }
    else
    {
        C->x = GB_array_xalloc (GB_SLOT (C, x), false, C_iso, nzmax,
            ctype->size, data_arena) ;
        if (C->x == NULL)
        { 
            // out of memory
            GB_FREE_ALL ;
            return (GrB_OUT_OF_MEMORY) ;
        }
        // C->x = (ctype) V->x
        GB_OK (GB_cast_matrix (C, V)) ;
    }

    //--------------------------------------------------------------------------
    // get the contents of C and determine # of threads to use
    //--------------------------------------------------------------------------

    int nthreads_max = GB_Context_nthreads_max ( ) ;
    double chunk = GB_Context_chunk ( ) ;
    int nthreads = GB_nthreads (vnz, chunk, nthreads_max) ;

    GB_Cp_DECLARE   (Cp, ) ; GB_Cp_PTR (Cp, C) ;
    GB_Ch_DECLARE   (Ch, ) ; GB_Ch_PTR (Ch, C) ;
    GB_Ci_DECLARE_U (Ci, ) ; GB_Ci_PTR (Ci, C) ;

    //--------------------------------------------------------------------------
    // construct the pattern of C: only the arrays not shared with V
    //--------------------------------------------------------------------------

    // A shared array also belongs to V and must not be written.  Each array
    // is filled by its own loop so that no loop branches per entry.
    const bool fill_h = C_is_hyper && !share_h ;
    const bool fill_i = !share_i ;
    int64_t p ;

    // Cp = 0..vnz-1
    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (p = 0 ; p < vnz ; p++)
    { 
        GB_ISET (Cp, p, p) ;            // Cp [p] = p ;
    }

    if (V_is_full)
    {

        //----------------------------------------------------------------------
        // V is full, or can be treated as full: entry p of V is in row p
        //----------------------------------------------------------------------

        if (fill_h)
        {
            #pragma omp parallel for num_threads(nthreads) schedule(static)
            for (p = 0 ; p < vnz ; p++)
            { 
                GB_ISET (Ch, p, p + kpositive) ;    // Ch [p] = p + kpositive
            }
        }

        if (fill_i)
        {
            #pragma omp parallel for num_threads(nthreads) schedule(static)
            for (p = 0 ; p < vnz ; p++)
            { 
                GB_ISET (Ci, p, p + knegative) ;    // Ci [p] = p + knegative
            }
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // V is sparse: entry p of V is in row Vi [p]
        //----------------------------------------------------------------------

        GB_Ai_DECLARE (Vi, const) ; GB_Ai_PTR (Vi, V) ;

        if (fill_h)
        {
            #pragma omp parallel for num_threads(nthreads) schedule(static)
            for (p = 0 ; p < vnz ; p++)
            { 
                int64_t i = GB_IGET (Vi, p) ;       // i = Vi [p]
                GB_ISET (Ch, p, i + kpositive) ;    // Ch [p] = i + kpositive
            }
        }

        if (fill_i)
        {
            #pragma omp parallel for num_threads(nthreads) schedule(static)
            for (p = 0 ; p < vnz ; p++)
            { 
                int64_t i = GB_IGET (Vi, p) ;       // i = Vi [p]
                GB_ISET (Ci, p, i + knegative) ;    // Ci [p] = i + knegative
            }
        }
    }

    //--------------------------------------------------------------------------
    // finalize the matrix C
    //--------------------------------------------------------------------------

    GB_ISET (Cp, vnz, vnz) ;    // Cp [vnz] = vnz ;
    C->nvals = vnz ;
    C->nvec = vnz ;
    GB_nvec_nonempty_set (C, vnz) ;
    C->magic = GB_MAGIC ;

    //--------------------------------------------------------------------------
    // free workspace, conform C to its desired format, and return result
    //--------------------------------------------------------------------------

    GB_FREE_WORKSPACE ;
    GB_OK (GB_conform (C, Werk)) ;
    ASSERT_MATRIX_OK (C, "C output for GB_Matrix_diag", GB0) ;
    return (GrB_SUCCESS) ;
}
