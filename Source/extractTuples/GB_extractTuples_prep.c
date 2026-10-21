//------------------------------------------------------------------------------
// GB_extractTuples_prep: prepare an output GrB_Vector for extractTuples
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// GB_extractTuples_prep ensures that an output GrB_Vector for I, J, or X is
// dense, of size nvals-by-1, and the right type.

#include "GB.h"
#include "extractTuples/GB_extractTuples.h"
#include "container/GB_container.h"

GrB_Info GB_extractTuples_prep
(
    GrB_Vector V,               // an output vector for I, J, or X
    uint64_t nvals,             // # of values V must hold
    const GrB_Type vtype        // desired type of V
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    if (V == NULL)
    { 
        // nothing to do; this component is not requested
        return (GrB_SUCCESS) ;
    }

    int data_arena = V->data_arena ;

    //--------------------------------------------------------------------------
    // quick return if V already has the right properties
    //--------------------------------------------------------------------------

    uint64_t required_memsize = nvals * vtype->size ;
    if (GB_IS_FULL (V) && V->nvals == nvals && V->vlen == nvals &&
        V->type == vtype && !(V->x_shallow) &&
        !GB_array_is_shared (GB_CSLOT (V, x)) &&
        GB_array_memsize (GB_CSLOT (V, x)) >= required_memsize)
    { 
        // nothing to do; the vector is already in the right format
        return (GrB_SUCCESS) ;
    }

    //--------------------------------------------------------------------------
    // remove V->x and free all other content
    //--------------------------------------------------------------------------

    // take V->x out of V (unless borrowed) so it is not freed, then free the
    // rest of V
    void *Vx = NULL ; uint64_t Vx_mem = 0 ; bool Vx_shallow = false ;
    if (!V->x_shallow)
    {
        GB_array_move (GB_VSLOT (Vx), GB_SLOT (V, x)) ;
    }
    GB_phybix_free ((GrB_Matrix) V) ;
    uint64_t Vx_memsize = GB_array_memsize (Vx, Vx_mem, Vx_shallow) ;

    //--------------------------------------------------------------------------
    // ensure Vx is large enough
    //--------------------------------------------------------------------------

    if (required_memsize > Vx_memsize || GB_array_is_shared (Vx, Vx_mem,
        Vx_shallow))
    {
        // If Vx is not large enough (or is shared with another matrix),
        // reallocate it.  If Vx was initially not empty, it means the space
        // is growing incrementally, so add 25% extra space for future growth.
        GB_array_release (GB_VSLOT (Vx)) ;
        int64_t n = nvals + ((Vx_memsize == 0) ? 0 : (nvals / 4)) ;
        Vx = GB_array_malloc (GB_VSLOT (Vx), n, vtype->size, data_arena) ;
        if (Vx == NULL)
        {
            return (GrB_OUT_OF_MEMORY) ;
        }
    }

    //--------------------------------------------------------------------------
    // load Vx back, to create V as a dense nvals-by-1 vector of type vtype
    //--------------------------------------------------------------------------

    GB_vector_load (V, GB_VSLOT (Vx), vtype, nvals) ;
    ASSERT_VECTOR_OK (V, "V prepped", GB0) ;
    return (GrB_SUCCESS) ;
}

