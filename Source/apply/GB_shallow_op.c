//------------------------------------------------------------------------------
// GB_shallow_op:  create a shallow copy and apply a unary operator to a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C = op (A), called only by GB_apply.

// Create a shallow copy of a matrix, applying an operator to the entries.

// The CSR/CSC format of C and A can differ, but they have they same vlen and
// vdim.  This function is CSR/CSC agnostic, except that C_is_csc is used to
// set the C->is_csc state in C.

// The values are typically not a shallow copy, unless no typecasting is needed
// and the operator is an identity operator.

// The pattern is always a shallow copy; C and A have the same integer sizes.
// Shallow matrices are never passed back to the user.

// Compare this function with GB_shallow_copy.c.

#include "apply/GB_apply.h"

#define GB_FREE_ALL GB_phybix_free (C) ;

GrB_Info GB_shallow_op      // create shallow matrix and apply operator
(
    GrB_Matrix C,           // output C, of type op*->ztype
    const bool C_is_csc,    // desired CSR/CSC format of C
        const GB_Operator op,       // unary/index-unary/binop to apply
        const GrB_Scalar scalar,    // scalar to bind to binary operator
        bool binop_bind1st,         // if true, binop(x,A) else binop(A,y)
        bool flipij,                // if true, flip i,j for user idxunop
    const GrB_Matrix A,     // input matrix to typecast
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (C != NULL) ;
    int header_arena = GB_arena (C->header_mem) ;
    int data_arena = C->data_arena ;

    ASSERT_MATRIX_OK (A, "A for shallow_op", GB0) ;
    ASSERT_OP_OK (op, "unop/binop for shallow_op", GB0) ;
    ASSERT (!GB_ZOMBIES (A)) ;
    ASSERT (GB_JUMBLED_OK (A)) ;
    ASSERT (!GB_PENDING (A)) ;

    GrB_Type ztype = op->ztype ;
    GrB_Type op_intype = NULL ;
    GB_Opcode opcode = op->opcode ;
    ASSERT (!GB_IS_INDEXBINARYOP_CODE (opcode)) ;

    if (GB_IS_UNARYOP_CODE (opcode))
    {
        ASSERT_OP_OK (op, "unop for shallow_op", GB0) ;
        if (!GB_IS_BUILTIN_UNOP_CODE_POSITIONAL (opcode))
        { 
            ASSERT (GB_Type_compatible (op->xtype, A->type)) ;
            op_intype = op->xtype ;
        }
    }
    else if (GB_IS_BINARYOP_CODE (opcode))
    {
        ASSERT_OP_OK (op, "binop for shallow_op", GB0) ;
        if (!GB_IS_BUILTIN_BINOP_CODE_POSITIONAL (opcode))
        { 
            op_intype = (binop_bind1st) ? op->ytype : op->xtype ;
            ASSERT (GB_Type_compatible (op_intype, A->type)) ;
        }
    }
    else // GB_IS_INDEXUNARYOP_CODE (opcode)
    { 
        ASSERT_OP_OK (op, "ixdunop for shallow_op", GB0) ;
        op_intype = op->xtype ;
        ASSERT (GB_Type_compatible (op_intype, A->type)) ;
    }

    //--------------------------------------------------------------------------
    // construct a shallow copy of A for the pattern of C
    //--------------------------------------------------------------------------

    GB_iso_code C_code_iso = GB_unop_code_iso (A, op, binop_bind1st) ;
    bool C_iso = (C_code_iso != GB_NON_ISO) ;

    // initialized the header for C, but do not allocate C->{p,h,b,i,x}
    // C has the exact same sparsity structure as A.
    GrB_Info info ;
    info = GB_new (&C, // any sparsity, existing header
        ztype, A->vlen, A->vdim, GB_ph_null, C_is_csc,
        GB_sparsity (A), A->hyper_switch, 0,
        A->p_is_32, A->j_is_32, A->i_is_32, header_arena, data_arena) ;
    ASSERT (info == GrB_SUCCESS) ;

    //--------------------------------------------------------------------------
    // make a shallow copy of the vector pointers
    //--------------------------------------------------------------------------

    // C->p and C->h are shared with A (or borrowed, if A borrows them)
    GB_array_share (GB_SLOT (C, p), GB_CSLOT (A, p)) ;    // size plen+1
    GB_array_share (GB_SLOT (C, h), GB_CSLOT (A, h)) ;    // size plen
    C->p_is_32 = A->p_is_32 ;
    C->j_is_32 = A->j_is_32 ;
    C->i_is_32 = A->i_is_32 ;
    C->plen = A->plen ;                 // C and A have the same hyperlist sizes
    C->nvec = A->nvec ;
    GB_nvec_nonempty_set (C, GB_nvec_nonempty_get (A)) ;
    C->jumbled = A->jumbled ;           // C is jumbled if A is jumbled
    C->nvals = A->nvals ;
    C->magic = GB_MAGIC ;
    C->iso = C_iso ;                    // OK
    if (C_iso)
    { 
        GB_BURBLE_MATRIX (A, "(iso apply) ") ;
    }

    //--------------------------------------------------------------------------
    // make a shallow copy of the A->Y hyper_hash
    //--------------------------------------------------------------------------

    C->Y = A->Y ;
    C->Y_shallow = (A->Y != NULL) ;
    C->no_hyper_hash = A->no_hyper_hash ;

    //--------------------------------------------------------------------------
    // check for empty matrix
    //--------------------------------------------------------------------------

    if (GB_nnz_max (A) == 0)
    { 
        // C->p and C->h are shallow but the rest is empty
        C->b = NULL ;
        C->i = NULL ;
        C->x = NULL ;
        C->b_shallow = false ;
        C->i_shallow = false ;
        C->x_shallow = false ;
        C->jumbled = false ;
        C->iso = false ;
        ASSERT_MATRIX_OK (C, "C = quick copy of empty A", GB0) ;
        return (GrB_SUCCESS) ;
    }

    //--------------------------------------------------------------------------
    // make a shallow copy of the pattern
    //--------------------------------------------------------------------------

    GB_array_share (GB_SLOT (C, b), GB_CSLOT (A, b)) ;
    GB_array_share (GB_SLOT (C, i), GB_CSLOT (A, i)) ;

    //--------------------------------------------------------------------------
    // make a shallow copy of the values, if possible
    //--------------------------------------------------------------------------

    // If the identity operator, first(A,y), or second(x,A) are used with no
    // typecasting, C->x becomes a shallow copy of A->x, and no work is done.

    int64_t anz = GB_nnz_held (A) ;

    if ((A->type == op_intype) &&
        ((opcode == GB_IDENTITY_unop_code) ||
         (opcode == GB_FIRST_binop_code  && !binop_bind1st) ||
         (opcode == GB_SECOND_binop_code &&  binop_bind1st)))
    { 
        // no work is done at all.  C is a pure shallow copy
        GBURBLE ("(pure shallow) ") ;
        // C->x is shared with A (or borrowed, if A borrows it)
        GB_array_share (GB_SLOT (C, x), GB_CSLOT (A, x)) ;
        C->iso = A->iso ;           // C has the same iso property as A
        ASSERT_MATRIX_OK (C, "C = pure shallow (A)", GB0) ;
        return (GrB_SUCCESS) ;
    }

    //--------------------------------------------------------------------------
    // apply the operator to the numerical values
    //--------------------------------------------------------------------------

    // allocate new space for the numerical values of C; use calloc if bitmap
    C->x = GB_array_xalloc (GB_SLOT (C, x), GB_IS_BITMAP (C), C_iso, anz,
        C->type->size, data_arena) ;    // C->x is freed when freeing C
    if (C->x == NULL)
    { 
        // out of memory
        GB_FREE_ALL ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    GB_OK (GB_apply_op ((GB_void *) C->x, data_arena, C->type, C_code_iso, op,
        scalar, binop_bind1st, flipij, A, Werk)) ;

    //--------------------------------------------------------------------------
    // return the result
    //--------------------------------------------------------------------------

    ASSERT_MATRIX_OK (C, "C = shallow (op (A))", GB0) ;
    return (GrB_SUCCESS) ;
}

