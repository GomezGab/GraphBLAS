//------------------------------------------------------------------------------
// GB_memoryUsage: # of bytes used for a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

void GB_memoryUsage         // count # allocated blocks and their sizes
(
    int64_t *nallocs,       // # of allocated memory blocks
    uint64_t *mem_deep,     // # of bytes in blocks owned by this matrix
    uint64_t *mem_shallow,  // # of bytes in blocks owned by another matrix
    const GrB_Matrix A,     // matrix to query
    bool count_hyper_hash   // if true, include A->Y
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (nallocs != NULL) ;
    ASSERT (mem_deep != NULL) ;
    ASSERT (mem_shallow != NULL) ;

    //--------------------------------------------------------------------------
    // count the allocated blocks and their sizes
    //--------------------------------------------------------------------------

    // a matrix contains 0 to 10 dynamically malloc'd blocks, not including
    // A->Y
    (*nallocs) = 0 ;
    (*mem_deep) = 0 ;
    (*mem_shallow) = 0 ;

    if (A == NULL)
    { 
        #pragma omp flush
        return ;
    }

    GB_Pending Pending = A->Pending ;

    (*nallocs)++ ;
    (*mem_deep) += GB_memsize (A->header_mem) ;

    if (A->p != NULL)
    {
        if (A->p_shallow || GB_array_is_shared (GB_CSLOT (A, p)))
        { 
            (*mem_shallow) += GB_array_memsize (GB_CSLOT (A, p)) ;
        }
        else
        { 
            (*nallocs)++ ;
            (*mem_deep) += GB_array_memsize (GB_CSLOT (A, p)) ;
        }
    }

    if (A->h != NULL)
    {
        if (A->h_shallow || GB_array_is_shared (GB_CSLOT (A, h)))
        { 
            (*mem_shallow) += GB_array_memsize (GB_CSLOT (A, h)) ;
        }
        else
        { 
            (*nallocs)++ ;
            (*mem_deep) += GB_array_memsize (GB_CSLOT (A, h)) ;
        }
    }

    if (A->b != NULL)
    {
        if (A->b_shallow || GB_array_is_shared (GB_CSLOT (A, b)))
        { 
            (*mem_shallow) += GB_array_memsize (GB_CSLOT (A, b)) ;
        }
        else
        { 
            (*nallocs)++ ;
            (*mem_deep) += GB_array_memsize (GB_CSLOT (A, b)) ;
        }
    }

    if (A->i != NULL)
    {
        if (A->i_shallow || GB_array_is_shared (GB_CSLOT (A, i)))
        { 
            (*mem_shallow) += GB_array_memsize (GB_CSLOT (A, i)) ;
        }
        else
        { 
            (*nallocs)++ ;
            (*mem_deep) += GB_array_memsize (GB_CSLOT (A, i)) ;
        }
    }

    if (A->x != NULL)
    {
        if (A->x_shallow || GB_array_is_shared (GB_CSLOT (A, x)))
        { 
            (*mem_shallow) += GB_array_memsize (GB_CSLOT (A, x)) ;
        }
        else
        { 
            (*nallocs)++ ;
            (*mem_deep) += GB_array_memsize (GB_CSLOT (A, x)) ;
        }
    }

    if (Pending != NULL)
    { 
        (*nallocs)++ ;
        (*mem_deep) += GB_memsize (Pending->header_mem) ;
    }

    if (Pending != NULL && Pending->i != NULL)
    { 
        (*nallocs)++ ;
        (*mem_deep) += GB_memsize (Pending->i_mem) ;
    }

    if (Pending != NULL && Pending->j != NULL)
    { 
        (*nallocs)++ ;
        (*mem_deep) += GB_memsize (Pending->j_mem) ;
    }

    if (Pending != NULL && Pending->x != NULL)
    { 
        (*nallocs)++ ;
        (*mem_deep) += GB_memsize (Pending->x_mem) ;
    }

    if (count_hyper_hash && A->Y != NULL)
    {
        int64_t Y_nallocs = 0 ;
        uint64_t Y_mem_deep = 0 ;
        uint64_t Y_mem_shallow = 0 ;
        GB_memoryUsage (&Y_nallocs, &Y_mem_deep, &Y_mem_shallow, A->Y, false) ;
        if (A->Y_shallow)
        { 
            // all of A->Y is shallow
            (*mem_shallow) += Y_mem_shallow + Y_mem_deep ;
        }
        else
        { 
            // A->Y itself is not shallow, but may contain shallow content
            (*nallocs) += Y_nallocs ;
            (*mem_deep) += Y_mem_deep ;
            (*mem_shallow) += Y_mem_shallow ;
        }
    }

    #pragma omp flush
    return ;
}

