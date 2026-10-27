//------------------------------------------------------------------------------
// GB_aliased.h: definitions for aliased and shallow matrices
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_ALIASED_H
#define GB_ALIASED_H

// GraphBLAS allows all inputs to all user-accessible objects to be aliased, as
// in GrB_mxm (C, C, accum, C, C, ...), which is valid.  Internal routines are
// more restrictive.

// GB_any_shared: true if A and B hold any array in common, whatever its
// ownership (same object, a borrowed view, or a reference-shared array).
// For method selection only; see GB_accum_mask.  A GB_same_pattern test
// (p, h, b, i only) would serve the structural-mask shortcuts; not yet needed.
bool GB_any_shared              // true if A and B hold any array in common
(
    const GrB_Matrix A,           // input A matrix
    const GrB_Matrix B            // input B matrix
) ;

bool GB_any_aliased             // determine if A and B are aliased in any way
(
    const GrB_Matrix A,           // input A matrix
    const GrB_Matrix B            // input B matrix
) ;

// GB_all_aliased is true if A and B are identical 
bool GB_all_aliased         // determine if A and B are all aliased
(
    const GrB_Matrix A,           // input A matrix
    const GrB_Matrix B            // input B matrix
) ;

// matrices returned to the user are never shallow; internal matrices may be
bool GB_is_shallow              // true if any component of A is shallow
(
    const GrB_Matrix A                // matrix to query
) ;

#endif

