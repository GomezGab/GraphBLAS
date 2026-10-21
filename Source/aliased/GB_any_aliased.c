//------------------------------------------------------------------------------
// GB_any_aliased: determine if two matrices are aliased in any way
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Returns true if A == B (and not NULL), or if any component A and B are
// aliased to each other.  In the latter case, that component of A and B will
// always be shallow, in either A or B, or both.  NULL pointers are not
// aliased.

#include "GB.h"

// true if pointers p1 and p2 are aliased and not NULL
// Two matrices are aliased if they are the same object, or if a component of
// one is a borrowed (shallow) view of the other.  Two matrices that merely
// share an owned, reference-counted array (after GrB_dup) are not aliased:
// the shared array is copied by GB_own_matrix before either is modified.
#define GB_POINTER_ALIASED(p1,p2) ((p1) == (p2) && (p1) != NULL)

bool GB_any_aliased             // determine if A and B are aliased
(
    GrB_Matrix A,           // input A matrix
    GrB_Matrix B            // input B matrix
)
{

    //--------------------------------------------------------------------------
    // check the matrices themselves
    //--------------------------------------------------------------------------

    if (A == NULL || B == NULL)
    { 
        // this is not an error condition; NULL pointers are not an alias
        return (false) ;
    }

    if (A == B)
    { 
        return (true) ;
    }

    //--------------------------------------------------------------------------
    // check their content
    //--------------------------------------------------------------------------

    bool aliased = false ;

    if (GB_POINTER_ALIASED (A->h, B->h) && (A->h_shallow || B->h_shallow))
    { 
        aliased = true ;
    }

    if (GB_POINTER_ALIASED (A->p, B->p) && (A->p_shallow || B->p_shallow))
    { 
        aliased = true ;
    }

    if (GB_POINTER_ALIASED (A->b, B->b) && (A->b_shallow || B->b_shallow))
    { 
        aliased = true ;
    }

    if (GB_POINTER_ALIASED (A->i, B->i) && (A->i_shallow || B->i_shallow))
    { 
        aliased = true ;
    }

    if (GB_POINTER_ALIASED (A->x, B->x) && (A->x_shallow || B->x_shallow))
    { 
        aliased = true ;
    }

    if (GB_any_aliased (A->Y, B->Y))
    { 
        // either Y is a shallow matrix, or two distinct Y matrices share arrays
        ASSERT (A->Y_shallow || B->Y_shallow || A->Y != B->Y) ;
        aliased = true ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    return (aliased) ;
}


//------------------------------------------------------------------------------
// GB_any_shared: determine if two matrices hold any array in common
//------------------------------------------------------------------------------

// Returns true if A == B, or if any of A->[phbix] is the same block as the
// corresponding B->[phbix], whether borrowed by one of them or shared by both
// through the reference-counted GB_array layer (after GrB_dup, GrB_apply,
// ...).  Unlike GB_any_aliased, this says nothing about whether an in-place
// write to one would corrupt the other (it would not for shared arrays, since
// GB_own_matrix copies them first).  It is for method selection: a shared
// array means the two matrices have the same content there, so a method that
// reads one while rewriting the other gains nothing over a method that builds
// the result from scratch, and may be much slower (see GB_accum_mask).

bool GB_any_shared              // true if A and B hold any array in common
(
    GrB_Matrix A,           // input A matrix
    GrB_Matrix B            // input B matrix
)
{
    if (A == NULL || B == NULL)
    { 
        return (false) ;
    }
    if (A == B)
    { 
        return (true) ;
    }
    return (GB_POINTER_ALIASED (A->h, B->h) || GB_POINTER_ALIASED (A->p, B->p)
         || GB_POINTER_ALIASED (A->b, B->b) || GB_POINTER_ALIASED (A->i, B->i)
         || GB_POINTER_ALIASED (A->x, B->x)) ;
}
