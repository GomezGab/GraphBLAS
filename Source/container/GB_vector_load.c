//------------------------------------------------------------------------------
// GB_vector_load: load C array into a dense GrB_Vector
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_container.h"

void GB_vector_load
(
    // input/output:
    GrB_Vector V,           // vector to load from the array slot X
    void **X,               // slot to move into V->x (see GB_array.h);
    uint64_t *X_mem,        // absent on output
    bool *X_shallow,
    // input:
    GrB_Type type,          // type of X
    uint64_t n              // # of entries in X
)
{

    //--------------------------------------------------------------------------
    // clear prior content of V and load X, making V a dense GrB_Vector
    //--------------------------------------------------------------------------

    // V->user_name is preserved; all other content is freed.  get/set controls
    // (hyper_switch, bitmap_switch, [pji]_control, etc) are preserved, except
    // that V->sparsity_control is revised to allow V to become a full vector.

    GB_phybix_free ((GrB_Matrix) V) ;

    V->type = type ;
    V->plen = -1 ;
    V->vlen = n ;
    V->vdim = 1 ;
    V->nvec = 1 ;
    GB_nvec_nonempty_set ((GrB_Matrix) V, (n == 0) ? 0 : 1) ;
    V->nvals = n ;
    V->sparsity_control = V->sparsity_control | GxB_FULL ;
    V->is_csc = true ;
    V->jumbled = false ;
    V->iso = false ;
    V->p_is_32 = false ;
    V->j_is_32 = false ;
    V->i_is_32 = false ;

    //--------------------------------------------------------------------------
    // load the content into V
    //--------------------------------------------------------------------------

    // move the slot X into V->x; X is absent on output.  A borrowed X
    // (readonly) remains borrowed in V, and is never freed by V.
    GB_array_move (GB_SLOT (V, x), X, X_mem, X_shallow) ;

    // V is now a valid GrB_Vector of length n, in the full format
    V->magic = GB_MAGIC ;
}

