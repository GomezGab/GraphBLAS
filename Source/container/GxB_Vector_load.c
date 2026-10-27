//------------------------------------------------------------------------------
// GxB_Vector_load: load C array into a dense GrB_Vector
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This method is guaranteed to take O(1) time and space.  If V starts as
// dense vector of length 0 with no content (V->x == NULL), then no malloc
// or frees are performed.

// If readonly is true, then V is created as a "shallow" vector.  Its
// numerical content, V->x = (*X), is "shallow" and thus treated as readonly
// by GraphBLAS.  It is not freed if V is freed with GrB_Vector_free.  (*X)
// need not be a malloc'd array at all.  Its allocation/deallocation is the
// responsibility of the user application.

// V is returned as a non-iso vector of length n, in the full data format.

// A load that follows a GxB_Vector_unload of the same vector, or that
// replaces content the vector owns exclusively, performs no allocation and
// cannot fail: the vector keeps the header of the unloaded (or replaced)
// array as a spare and reuses it (see GB_array_spare_take in
// memory/GB_array.c).  A readonly load never allocates.  Only a load with
// GrB_DEFAULT handling into a vector that has neither a spare nor exclusively
// owned content allocates a header, and only that case can return
// GrB_OUT_OF_MEMORY; *X then remains the caller's.

// If handling is GxB_IS_READONLY, *X is returned unchanged.  Otherwise, it is
// returned as NULL to indicate that it has been moved into V.

// The vector V may have readonly components on input; they are simply removed
// from V and not modified.

// The handling parameter can be one of:
//
//      GrB_DEFAULT + data_arena
//      GxB_IS_READONLY + data_arena

#include "GB_container.h"

GrB_Info GxB_Vector_load
(
    // input/output:
    GrB_Vector V,           // vector to load from the C array X
    void **X,               // numerical array to load into V
    // input:
    GrB_Type type,          // type of X
    uint64_t n,             // # of entries in X
    uint64_t X_memsize,     // size of X in bytes (at least n*(sizeof the type))
    int handling,           // GrB_DEFAULT (0): transfer ownership to GraphBLAS
                            // GxB_IS_READONLY: X treated as readonly;
                            //  ownership kept by the user application
    const GrB_Descriptor desc   // currently unused; for future expansion
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_CHECK_INIT ;
    GB_RETURN_IF_NULL_OR_FAULTY (V) ;
    GB_RETURN_IF_NULL_OR_FAULTY (type) ;
    GB_RETURN_IF_NULL (X) ;
    if (n > 0)
    { 
        GB_RETURN_IF_NULL (*X) ;
    }
    if (X_memsize < n * type->size)
    { 
        // X is too small
        return (GrB_INVALID_VALUE) ;
    }
    ASSERT_VECTOR_OK (V, "V to load (contents mostly ignored)", GB0) ;

    int data_arena ;
    bool readonly ;

    if (handling >= ((int) GrB_DEFAULT) &&
        handling <  ((int) GrB_DEFAULT) + GxB_NARENAS)
    { 
        data_arena = handling - ((int) GrB_DEFAULT) ;
        readonly = false ;
    }
    else if (handling >= ((int) GxB_IS_READONLY) &&
             handling <  ((int) GxB_IS_READONLY) + GxB_NARENAS)
    { 
        data_arena = handling - ((int) GxB_IS_READONLY) ;
        readonly = true ;
    }
    else
    { 
        // invalid handling
        return (GrB_INVALID_VALUE) ;
    }

    if (GB_Global_malloc_function_get (data_arena) == NULL)
    { 
        // arena out of range or not initialized
        return (GrB_INVALID_VALUE) ;
    }

    uint64_t X_mem = GB_mem (data_arena, X_memsize) ;

    //--------------------------------------------------------------------------
    // clear prior content of V and load X, making V a dense GrB_Vector
    //--------------------------------------------------------------------------

    // V->user_name is preserved; all other content is freed.  get/set controls
    // (hyper_switch, bitmap_switch, [pji]_control, etc) are preserved, except
    // that V->sparsity_control is revised to allow V to become a full vector.

    // Place *X in an array slot: borrowed if readonly, else owned by GraphBLAS
    // (and added to the global debug memtable).  The header left behind by a
    // prior GxB_Vector_unload of V, or the header of the content of V that
    // is about to be discarded, is reused, so that this load cannot fail in
    // those cases.  Otherwise, if a header cannot be allocated, *X remains the
    // property of the user application and V is unchanged.
    GB_array_hdr *spare = GB_array_spare_take (GB_SLOT (V, x)) ;
    GB_VSLOT_DECLARE (void *, Xs) ;
    bool ok ;
    GB_array_load (GB_VSLOT (Xs), *X, X_mem, readonly, true, spare, &ok) ;
    if (!ok)
    {
        // out of memory
        return (GrB_OUT_OF_MEMORY) ;
    }
    if (!readonly)
    {
        // tell the caller that X has been moved into V
        (*X) = NULL ;
    }

    GB_vector_load (V, GB_VSLOT (Xs), type, n) ;

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    ASSERT_VECTOR_OK (V, "V loaded", GB0) ;
    ASSERT (GB_IS_FULL (V)) ;
    return (GrB_SUCCESS) ;
}

