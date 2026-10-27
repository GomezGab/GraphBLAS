//------------------------------------------------------------------------------
// GB_convert_int: convert the integers in a matrix to/from 32/64 bits
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The integer arrays A->[phi] and A->Y in the matrix A are converted to match
// the requested p_is_32_new, j_is_32_new, and i_is_32_new.  If converted,
// A->[phi] are no longer shallow.  If A->Y is entirely shallow, it is simply
// removed from A.  If A->Y is itself not shallow but contains any shallow
// A->Y->[phi] components, those components are converted and are no longer
// shallow.

// If A has too many entries for p_is_32_new == true, A->p is left unchanged.
// If the dimension of A is too large for j_is_32_new == true, A->h and A->Y
// are left unchanged.  If the dimension of A is too large for i_is_32_new ==
// true, A->i.  is left unchanged.  These are not error conditions.

#include "GB.h"
#define GB_FREE_ALL ;

GrB_Info GB_convert_int     // convert the integers of a matrix
(
    GrB_Matrix A,           // matrix to convert
    bool p_is_32_new,       // new integer format for A->p
    bool j_is_32_new,       // new integer format for A->p
    bool i_is_32_new,       // new integer format for A->h, A->i, and A->Y
    bool determine          // if true, revise p_is_32_new, j_is_32_new,
        // and i_is_32_new based on A->vlen, A->vdim and A->nvals.  Otherwise,
        // ignore the matrix properties and always convert to the new integer
        // sizes.
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    if (A == NULL || A->magic != GB_MAGIC)
    { 
        // nothing to convert 
        return (GrB_SUCCESS) ;
    }

    ASSERT (GB_ZOMBIES_OK (A)) ;
    ASSERT (GB_JUMBLED_OK (A)) ;
    ASSERT (GB_PENDING_OK (A)) ;

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    if (GB_IS_FULL (A) || GB_IS_BITMAP (A))
    { 
        // quick return: nothing to do
        return (GrB_SUCCESS) ;
    }

    int header_arena = GB_arena (A->header_mem) ;
    int data_arena = A->data_arena ;
    uint64_t mem = GB_mem (data_arena, 0) ;

    int64_t anz = GB_nnz (A) ;
    if (determine)
    {
        ASSERT_MATRIX_OK (A, "A converting integers", GB0) ;
        p_is_32_new = GB_determine_p_is_32 (p_is_32_new, anz) ;
        j_is_32_new = GB_determine_j_is_32 (j_is_32_new, A->vdim) ;
        i_is_32_new = GB_determine_i_is_32 (i_is_32_new, A->vlen) ;
    }
    bool p_is_32 = A->p_is_32 ;
    bool j_is_32 = A->j_is_32 ;
    bool i_is_32 = A->i_is_32 ;

    if (p_is_32 == p_is_32_new &&
        j_is_32 == j_is_32_new &&
        i_is_32 == i_is_32_new)
    { 
        // quick return: nothing to do
        return (GrB_SUCCESS) ;
    }

    //--------------------------------------------------------------------------
    // at least some integers must be converted
    //--------------------------------------------------------------------------

    // simply remove A->Y if it is entirely shallow
    if (A->Y_shallow)
    { 
        A->Y = NULL ;
        A->Y_shallow = false ;
    }

    bool A_is_hyper = GB_IS_HYPERSPARSE (A) ;   // (A->h != NULL)
    int64_t plen = A->plen ;
    GrB_Matrix Y = A->Y ;
    GB_Pending Pending = A->Pending ;
    int64_t ynz = GB_nnz (Y) ;
    int64_t yplen = (Y == NULL) ? 0 : Y->plen ;
    int64_t npending = (Pending == NULL) ? 0 : Pending->n ;
    int64_t nmax_pending = (Pending == NULL) ? 0 : Pending->nmax ;

    //--------------------------------------------------------------------------
    // allocate new space for A->[phi] and Y->[pix] if present
    //--------------------------------------------------------------------------

    // Y is not converted via a recursive call to this method.  Instead, it is
    // converted directly below.  This is because Y->x must also be converted,
    // and also so that the conversion will be all-or-nothing, if out of
    // memory.

    GB_VSLOT_DECLARE (void *, Ap_new) ;
    GB_VSLOT_DECLARE (void *, Ah_new) ;
    GB_VSLOT_DECLARE (void *, Ai_new) ;
    GB_VSLOT_DECLARE (void *, Yp_new) ;
    GB_VSLOT_DECLARE (void *, Yi_new) ;
    GB_VSLOT_DECLARE (void *, Yx_new) ;
    void *Pending_i_new = NULL ; uint64_t Pending_i_new_mem = mem ;
    void *Pending_j_new = NULL ; uint64_t Pending_j_new_mem = mem ;
    bool has_Pending_i = (Pending != NULL) && (Pending->i != NULL) ;
    bool has_Pending_j = (Pending != NULL) && (Pending->j != NULL) ;

    size_t psize_new = p_is_32_new ? sizeof (uint32_t) : sizeof (uint64_t) ;
    size_t jsize_new = j_is_32_new ? sizeof (uint32_t) : sizeof (uint64_t) ;
    size_t isize_new = i_is_32_new ? sizeof (uint32_t) : sizeof (uint64_t) ;

    bool ok = true ;

    if (p_is_32 != p_is_32_new && A->p != NULL)
    { 
        // allocate new space for A->p
        Ap_new = GB_array_malloc (GB_VSLOT (Ap_new), plen+1, psize_new,
            data_arena) ;
        ok = ok && (Ap_new != NULL) ;
    }

    if (i_is_32 != i_is_32_new)
    { 
        // allocate new space for A->i
        if (A->i != NULL)
        {
            Ai_new = GB_array_malloc (GB_VSLOT (Ai_new), anz, isize_new,
                data_arena) ;
            ok = ok && (Ai_new != NULL) ;
        }
        if (has_Pending_i)
        {
            // allocate new space for Pending->i; matches A->i_is_32
            Pending_i_new = GB_MALLOC_MEMORY (nmax_pending, isize_new,
                &Pending_i_new_mem) ;
            ok = ok && (Pending_i_new != NULL) ;
        }
    }

    if (j_is_32 != j_is_32_new)
    {
        if (A_is_hyper)
        { 
            // allocate new space for A->h
            Ah_new = GB_array_malloc (GB_VSLOT (Ah_new), plen, jsize_new,
                data_arena) ;
            ok = ok && (Ah_new != NULL) ;
        }
        if (Y != NULL)
        { 
            // allocate new space for Y->[phi]; matches A->j_is_32
            Yp_new = GB_array_malloc (GB_VSLOT (Yp_new), yplen+1, jsize_new,
                data_arena) ;
            Yi_new = GB_array_malloc (GB_VSLOT (Yi_new), ynz,     jsize_new,
                data_arena) ;
            Yx_new = GB_array_malloc (GB_VSLOT (Yx_new), ynz,     jsize_new,
                data_arena) ;
            ok = ok && (Yp_new != NULL && Yi_new != NULL && Yx_new != NULL) ;
        }
        if (has_Pending_j)
        { 
            // allocate new space for Pending->j; matches A->j_is_32
            Pending_j_new = GB_MALLOC_MEMORY (nmax_pending, jsize_new,
                &Pending_j_new_mem) ;
            ok = ok && (Pending_j_new != NULL) ;
        }
    }

    if (!ok)
    { 
        // out of memory: A is unchanged
        GB_array_release (GB_VSLOT (Ap_new)) ;
        GB_array_release (GB_VSLOT (Ah_new)) ;
        GB_array_release (GB_VSLOT (Ai_new)) ;
        GB_array_release (GB_VSLOT (Yp_new)) ;
        GB_array_release (GB_VSLOT (Yi_new)) ;
        GB_array_release (GB_VSLOT (Yx_new)) ;
        GB_FREE_MEMORY (&Pending_i_new, Pending_i_new_mem) ;
        GB_FREE_MEMORY (&Pending_j_new, Pending_j_new_mem) ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    // the conversion will now succeed

    //--------------------------------------------------------------------------
    // convert A->p
    //--------------------------------------------------------------------------

    int nthreads_max = GB_Context_nthreads_max ( ) ;

    if (p_is_32 != p_is_32_new && A->p != NULL)
    { 
        GB_cast_int (Ap_new, p_is_32_new ? GB_UINT32_code : GB_UINT64_code,
                     A->p  , p_is_32     ? GB_UINT32_code : GB_UINT64_code,
                     plen+1, nthreads_max) ;
        GB_array_release (GB_SLOT (A, p)) ;
        GB_array_move (GB_SLOT (A, p), GB_VSLOT (Ap_new)) ;
    }

    A->p_is_32 = p_is_32_new ;

    //--------------------------------------------------------------------------
    // convert A->i and Pending->i
    //--------------------------------------------------------------------------

    if (i_is_32 != i_is_32_new)
    { 
        bool zombies = (A->nzombies > 0) ;
        GB_Type_code zombie32  = zombies     ? GB_INT32_code  : GB_UINT32_code ;
        GB_Type_code zombie64  = zombies     ? GB_INT32_code  : GB_UINT64_code ;
        GB_Type_code icode_new = i_is_32_new ? zombie32       : zombie64       ;
        GB_Type_code icode     = i_is_32     ? zombie32       : zombie64       ;
        GB_Type_code ucode_new = i_is_32_new ? GB_UINT32_code : GB_UINT64_code ;
        GB_Type_code ucode     = i_is_32     ? GB_UINT32_code : GB_UINT64_code ;

        //----------------------------------------------------------------------
        // convert A->i
        //----------------------------------------------------------------------

        if (A->i != NULL)
        { 
            GB_cast_int (Ai_new, icode_new, A->i, icode, anz, nthreads_max) ;
            GB_array_release (GB_SLOT (A, i)) ;
            GB_array_move (GB_SLOT (A, i), GB_VSLOT (Ai_new)) ;
        }

        //----------------------------------------------------------------------
        // convert Pending->i if present
        //----------------------------------------------------------------------

        if (has_Pending_i)
        { 
            GB_cast_int (Pending_i_new, ucode_new, Pending->i, ucode, npending,
                nthreads_max) ;
            GB_FREE_MEMORY (&(Pending->i), Pending->i_mem) ;
            Pending->i = Pending_i_new ;
            Pending->i_mem = Pending_i_new_mem ;
        }
    }

    A->i_is_32 = i_is_32_new ;

    //--------------------------------------------------------------------------
    // convert A->h, Y->p, Y->i, Y->x, and Pending->j
    //--------------------------------------------------------------------------

    if (j_is_32 != j_is_32_new)
    {

        GB_Type_code ucode_new = j_is_32_new ? GB_UINT32_code : GB_UINT64_code ;
        GB_Type_code ucode     = j_is_32     ? GB_UINT32_code : GB_UINT64_code ;

        //----------------------------------------------------------------------
        // convert A->h if present
        //----------------------------------------------------------------------

        if (A_is_hyper)
        { 
            GB_cast_int (Ah_new, ucode_new, A->h, ucode, plen, nthreads_max) ;
            GB_array_release (GB_SLOT (A, h)) ;
            GB_array_move (GB_SLOT (A, h), GB_VSLOT (Ah_new)) ;
        }

        //----------------------------------------------------------------------
        // convert A->Y if present
        //----------------------------------------------------------------------

        if (Y != NULL)
        { 
            // A is hypersparse, and the integers of Y match A->j_is_32
            ASSERT (A_is_hyper) ;
            ASSERT (Y->p_is_32 == j_is_32) ;
            ASSERT (Y->j_is_32 == j_is_32) ;
            ASSERT (Y->i_is_32 == j_is_32) ;
            ASSERT_MATRIX_OK (Y, "Y converting integers", GB0) ;

            //------------------------------------------------------------------
            // convert Y->p
            //------------------------------------------------------------------

            GB_cast_int (Yp_new, ucode_new, Y->p, ucode, yplen+1, nthreads_max);
            GB_array_release (GB_SLOT (Y, p)) ;
            GB_array_move (GB_SLOT (Y, p), GB_VSLOT (Yp_new)) ;
            Y->p_is_32 = j_is_32_new ;

            //------------------------------------------------------------------
            // convert Y->i
            //------------------------------------------------------------------

            GB_cast_int (Yi_new, ucode_new, Y->i, ucode, ynz, nthreads_max) ;
            GB_array_release (GB_SLOT (Y, i)) ;
            GB_array_move (GB_SLOT (Y, i), GB_VSLOT (Yi_new)) ;
            Y->i_is_32 = j_is_32_new ;

            //------------------------------------------------------------------
            // convert Y->x
            //------------------------------------------------------------------

            GB_cast_int (Yx_new, ucode_new, Y->x, ucode, ynz, nthreads_max) ;
            GB_array_release (GB_SLOT (Y, x)) ;
            GB_array_move (GB_SLOT (Y, x), GB_VSLOT (Yx_new)) ;
            Y->type = j_is_32_new ? GrB_UINT32 : GrB_UINT64 ;

            //------------------------------------------------------------------
            // revise Y->j_is_32
            //------------------------------------------------------------------

            Y->j_is_32 = j_is_32_new ;
            ASSERT_MATRIX_OK (Y, "Y converted integers", GB0) ;
        }

        //----------------------------------------------------------------------
        // convert Pending->j if present
        //----------------------------------------------------------------------

        if (has_Pending_j)
        { 
            GB_cast_int (Pending_j_new, ucode_new, Pending->j, ucode, npending,
                nthreads_max) ;
            GB_FREE_MEMORY (&(Pending->j), Pending->j_mem) ;
            Pending->j = Pending_j_new ;
            Pending->j_mem = Pending_j_new_mem ;
        }
    }

    A->j_is_32 = j_is_32_new ;

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    ASSERT_MATRIX_OK (A, "A integers converted", GB0) ;
    return (GrB_SUCCESS) ;
}

