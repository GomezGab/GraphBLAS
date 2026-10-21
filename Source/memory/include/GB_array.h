//------------------------------------------------------------------------------
// GB_array.h: reference-counted ownership of the A->[phbix] arrays of a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// A matrix holds up to five arrays: A->p, A->h, A->b, A->i, and A->x.  Each
// array is described by three fields of the GB_Matrix_opaque struct, which
// together form a "slot":
//
//      void *A->c ;            // the raw pointer (int8_t * for A->b)
//      uint64_t A->c_mem ;     // see below
//      bool A->c_shallow ;     // true if the array is borrowed
//
// where c is one of p, h, b, i, or x.  The layout of GB_Matrix_opaque does not
// change; only the meaning of A->c_mem does.  A slot is in one of three
// states:
//
//      absent:     A->c == NULL, A->c_shallow == false, A->c_mem == 0.
//
//      borrowed:   A->c_shallow == true.  A->c points into memory that this
//                  matrix does not own (another matrix, a user-provided
//                  read-only array, or a static scalar).  A->c_mem holds the
//                  plain GB_mem (arena, memsize) word, exactly as before.
//                  The array is never freed or written through this slot.
//
//      owned:      A->c_shallow == false and A->c != NULL.  A->c is the
//                  pointer returned by GB_malloc_memory, so it can be handed
//                  to the user application by GxB_unload and friends.
//                  A->c_mem holds the address of a GB_array_hdr, cast to
//                  uint64_t, which carries the GB_mem word of the block and
//                  the number of matrices that own it (hdr->refs).  The block
//                  is shared iff hdr->refs > 1, and must then be treated as
//                  read-only until GB_array_own is called on the slot.
//
// All memory traffic for the five arrays goes through this layer; nothing
// above it calls GB_malloc_memory / GB_free_memory for them directly.  The
// raw pointer A->c is unchanged, so all kernels read it as before.
//
// Thread safety: hdr->refs is only ever modified with the GB_ATOMIC_FETCH_*
// macros.  A share only happens on an input matrix of an operation (by a
// thread that already holds a reference), and an own or release only happens
// on an output matrix.  A thread that finds refs > 1 in GB_array_own copies
// the block before it decrements refs, so no thread ever writes a block that
// another thread may still be reading.

#ifndef GB_ARRAY_H
#define GB_ARRAY_H

//------------------------------------------------------------------------------
// GB_array_hdr: the header of an owned block
//------------------------------------------------------------------------------

typedef struct GB_array_hdr_struct
{
    uint64_t mem ;          // memsize and arena of the block: GB_mem encoding
    uint64_t hdr_mem ;      // memsize and arena of this header itself
    int64_t refs ;          // # of slots that own the block; always >= 1
}
GB_array_hdr ;

//------------------------------------------------------------------------------
// GB_SLOT: expand the three fields of a slot
//------------------------------------------------------------------------------

// Usage: GB_array_release (GB_SLOT (A, x)) ;
// The cast is required since A->b is int8_t *, not void *.

#define GB_SLOT(A,c) \
    (void **) &((A)->c), &((A)->c ## _mem), &((A)->c ## _shallow)

// GB_CSLOT: the same, for read-only queries that take values, not pointers
#define GB_CSLOT(A,c) \
    (const void *) ((A)->c), ((A)->c ## _mem), ((A)->c ## _shallow)

// GB_VSLOT: a slot held in local variables v, v_mem, and v_shallow, for
// workspace that will be moved into a matrix with GB_array_move.  Declare as:
//      type *v = NULL ; uint64_t v_mem = 0 ; bool v_shallow = false ;
// allocate with GB_array_malloc (GB_VSLOT (v), ...), free on error with
// GB_array_release (GB_VSLOT (v)), and GB_array_move (GB_SLOT (A, c),
// GB_VSLOT (v)) cannot fail, so no new error path is needed at the move.
#define GB_VSLOT(v) \
    (void **) &(v), &(v ## _mem), &(v ## _shallow)

//------------------------------------------------------------------------------
// GB_array_hdr_get: the header of an owned slot, or NULL
//------------------------------------------------------------------------------

GB_STATIC_INLINE GB_array_hdr *GB_array_hdr_get
(
    const void *p,          // raw pointer of the slot
    uint64_t mem,           // A->c_mem field
    bool shallow            // A->c_shallow field
)
{
    if (p == NULL || shallow)
    {
        // absent or borrowed: no header
        return (NULL) ;
    }
    return ((GB_array_hdr *) (uintptr_t) mem) ;
}

//------------------------------------------------------------------------------
// queries: valid for all three states
//------------------------------------------------------------------------------

// GB_array_mem: the plain GB_mem (arena, memsize) word of the block, or 0 if
// the slot is absent.  This replaces every read of A->c_mem that wants the
// memsize or arena, including the values returned to the user by GxB_unload,
// GxB_export, GxB_unpack, and GxB_Matrix_memoryUsage.

GB_STATIC_INLINE uint64_t GB_array_mem
(
    const void *p,
    uint64_t mem,
    bool shallow
)
{
    if (p == NULL)
    {
        return (0) ;
    }
    if (shallow)
    {
        // borrowed: the field is the plain word
        return (mem) ;
    }
    // owned: the field is the header address
    return (((GB_array_hdr *) (uintptr_t) mem)->mem) ;
}

GB_STATIC_INLINE uint64_t GB_array_memsize (const void *p, uint64_t mem,
    bool shallow)
{
    return (GB_memsize (GB_array_mem (p, mem, shallow))) ;
}

GB_STATIC_INLINE int GB_array_arena (const void *p, uint64_t mem, bool shallow)
{
    return (GB_arena (GB_array_mem (p, mem, shallow))) ;
}

// GB_array_refs: # of owners of the block, or 0 if absent or borrowed
GB_STATIC_INLINE int64_t GB_array_refs (const void *p, uint64_t mem,
    bool shallow)
{
    GB_array_hdr *hdr = GB_array_hdr_get (p, mem, shallow) ;
    if (hdr == NULL)
    {
        return (0) ;
    }
    int64_t refs ;
    GB_ATOMIC_LOAD64 (refs, hdr->refs) ;
    return (refs) ;
}

// GB_array_is_shared: true if the block is owned by more than one slot, and
// thus must not be written until GB_array_own is called
GB_STATIC_INLINE bool GB_array_is_shared (const void *p, uint64_t mem,
    bool shallow)
{
    return (GB_array_refs (p, mem, shallow) > 1) ;
}

//------------------------------------------------------------------------------
// GB_array_clear: set a slot to the absent state without freeing anything
//------------------------------------------------------------------------------

GB_STATIC_INLINE void GB_array_clear (void **p, uint64_t *mem, bool *shallow)
{
    (*p) = NULL ;
    (*mem) = 0 ;
    (*shallow) = false ;
}

//------------------------------------------------------------------------------
// spare headers
//------------------------------------------------------------------------------

// An absent slot (p == NULL, shallow == false) may hold a spare header in its
// mem word: a header with refs == 0 and no block.  GxB_Vector_unload leaves
// one behind so that a following GxB_Vector_load into the same vector needs
// no allocation, and thus cannot fail, as it could not before headers
// existed.  Every operation that fills or releases a slot frees a spare it
// finds, so a spare never leaks.

// A spare is tagged by setting bit 0 of the mem word (headers are 8-byte
// aligned), so a stale header address or plain mem word left in an absent
// slot is never mistaken for a spare.
#define GB_SPARE_TAG ((uint64_t) 1)

GB_STATIC_INLINE GB_array_hdr *GB_array_spare_get (const void *p, uint64_t mem,
    bool shallow)
{
    return ((p == NULL && !shallow && (mem & GB_SPARE_TAG)) ?
        (GB_array_hdr *) (uintptr_t) (mem & ~GB_SPARE_TAG) : NULL) ;
}

GB_STATIC_INLINE void GB_array_spare_free (void **p, uint64_t *mem,
    bool *shallow)
{
    GB_array_hdr *spare = GB_array_spare_get (*p, *mem, *shallow) ;
    if (spare != NULL)
    {
        uint64_t hdr_mem = spare->hdr_mem ;
        GB_FREE_MEMORY (&spare, hdr_mem) ;
        (*mem) = 0 ;
    }
}

//------------------------------------------------------------------------------
// GB_array_borrow: make a slot a borrowed view of a block we do not own
//------------------------------------------------------------------------------

// O(1), cannot fail.  Used for user read-only arrays (GxB_IS_READONLY), static
// scalars (GB_Scalar_wrap), and shallow views of another matrix.  The slot
// must be absent on input (release it first).

GB_STATIC_INLINE void GB_array_borrow
(
    void **p, uint64_t *mem, bool *shallow,     // slot to set
    void *block,                                // block to borrow
    uint64_t block_mem                          // plain GB_mem word of block
)
{
    ASSERT ((*p) == NULL) ;
    GB_array_spare_free (p, mem, shallow) ;
    if (block == NULL)
    {
        GB_array_clear (p, mem, shallow) ;
    }
    else
    {
        (*p) = block ;
        (*mem) = block_mem ;
        (*shallow) = true ;
    }
}

// GB_array_borrow_from: C->c becomes a borrowed view of A->c.  This copies
// the plain mem word of A->c, never the address of its header, so C is always
// safe to free with GB_array_release regardless of the state of A.

GB_STATIC_INLINE void GB_array_borrow_from
(
    void **cp, uint64_t *cmem, bool *cshallow,              // slot of C
    const void *ap, uint64_t amem, bool ashallow            // slot of A
)
{
    GB_array_borrow (cp, cmem, cshallow, (void *) ap,
        GB_array_mem (ap, amem, ashallow)) ;
}

//------------------------------------------------------------------------------
// GB_array_move: transplant a slot from A to C
//------------------------------------------------------------------------------

// C takes the pointer, the header (or the borrowed state), and A becomes
// absent.  O(1), cannot fail.  C must be absent on input.

GB_STATIC_INLINE void GB_array_move
(
    void **cp, uint64_t *cmem, bool *cshallow,      // slot of C: output
    void **ap, uint64_t *amem, bool *ashallow       // slot of A: cleared
)
{
    ASSERT ((*cp) == NULL) ;
    GB_array_spare_free (cp, cmem, cshallow) ;
    (*cp) = (*ap) ;
    (*cmem) = (*amem) ;
    (*cshallow) = (*ashallow) ;
    GB_array_clear (ap, amem, ashallow) ;
}

//------------------------------------------------------------------------------
// GB_array_share: C->c becomes a co-owner of the block owned by A->c
//------------------------------------------------------------------------------

// O(1), cannot fail.  C must be absent on input.  If A->c is borrowed, C->c
// is also borrowed (there is nothing to own).  The caller must hold a
// reference to A for the duration of the call (it is reading A).

GB_STATIC_INLINE void GB_array_share
(
    void **cp, uint64_t *cmem, bool *cshallow,              // slot of C
    const void *ap, uint64_t amem, bool ashallow            // slot of A
)
{
    ASSERT ((*cp) == NULL) ;
    GB_array_spare_free (cp, cmem, cshallow) ;
    GB_array_hdr *hdr = GB_array_hdr_get (ap, amem, ashallow) ;
    if (hdr == NULL)
    {
        // A->c is absent or borrowed
        GB_array_borrow_from (cp, cmem, cshallow, ap, amem, ashallow) ;
    }
    else
    {
        int64_t old_refs ;
        GB_ATOMIC_FETCH_INC64 (old_refs, hdr->refs) ;
        ASSERT (old_refs >= 1) ;
        (void) old_refs ;
        (*cp) = (void *) ap ;
        (*cmem) = amem ;
        (*cshallow) = false ;
    }
}

//------------------------------------------------------------------------------
// GB_array_release: drop a slot's reference to its block
//------------------------------------------------------------------------------

// absent: nothing to do.  borrowed: forget the pointer.  owned: decrement the
// reference count, and free the block and its header if this was the last
// reference.  The slot is absent on output.  Replaces the sequence:
//
//      if (!A->c_shallow) GB_FREE_MEMORY (&(A->c), A->c_mem) ;
//      A->c = NULL ; A->c_mem = 0 ; A->c_shallow = false ;

GB_STATIC_INLINE void GB_array_release (void **p, uint64_t *mem, bool *shallow)
{
    GB_array_spare_free (p, mem, shallow) ;
    GB_array_hdr *hdr = GB_array_hdr_get (*p, *mem, *shallow) ;
    if (hdr != NULL)
    {
        int64_t old_refs ;
        GB_ATOMIC_FETCH_DEC64 (old_refs, hdr->refs) ;
        ASSERT (old_refs >= 1) ;
        if (old_refs == 1)
        {
            // this was the last reference: free the block and the header
            uint64_t block_mem = hdr->mem ;
            uint64_t hdr_mem = hdr->hdr_mem ;
            GB_FREE_MEMORY (p, block_mem) ;
            GB_FREE_MEMORY (&hdr, hdr_mem) ;
        }
    }
    GB_array_clear (p, mem, shallow) ;
}

//------------------------------------------------------------------------------
// non-inline methods: see memory/GB_array.c
//------------------------------------------------------------------------------

GB_array_hdr *GB_array_hdr_new      // a new header (refs 1), or NULL if failed
(
    uint64_t block_mem      // memsize and arena of the block it will own; the
                            // header is allocated in that arena
) ;

GrB_Info GB_array_adopt     // GrB_SUCCESS, or GrB_OUT_OF_MEMORY
(
    // input/output:
    void **p, uint64_t *mem, bool *shallow,     // slot to set; absent on input
    // input:
    void *block,            // block from GB_malloc_memory, or NULL
    uint64_t block_mem      // its memsize and arena, from GB_malloc_memory
) ;

void *GB_array_malloc       // pointer to the new block, or NULL if failed
(
    // input/output:
    void **p, uint64_t *mem, bool *shallow,     // slot to set; absent on input
    // input:
    uint64_t nitems,        // # of items to allocate
    uint64_t size_of_item,  // size of each item
    int arena               // arena to allocate the block and header in
) ;

void *GB_array_calloc       // pointer to the new block, or NULL if failed
(
    // input/output:
    void **p, uint64_t *mem, bool *shallow,     // slot to set; absent on input
    // input:
    uint64_t nitems,        // # of items to allocate
    uint64_t size_of_item,  // size of each item
    int arena               // arena to allocate the block and header in
) ;

void *GB_array_xalloc       // pointer to the new block, or NULL if failed
(
    // input/output:
    void **p, uint64_t *mem, bool *shallow,     // slot to set; absent on input
    // input:
    bool use_calloc,        // if true, use calloc
    bool iso,               // if true, only allocate a single entry
    uint64_t nentries,      // # of entries to allocate if non iso
    uint64_t sizeof_entry,  // size of each entry
    int arena               // arena to allocate the block and header in
) ;

GrB_Info GB_array_own       // GrB_SUCCESS or GrB_OUT_OF_MEMORY
(
    // input/output:
    void **p, uint64_t *mem, bool *shallow,     // slot to make writable
    // input:
    uint64_t nbytes,        // # of bytes of the block that are live
    int arena,              // arena for the copy, if one is needed
    int nthreads            // # of threads to use for the copy
) ;

// which arrays GB_own_matrix must make private
#define GB_OWN_P   1
#define GB_OWN_H   2
#define GB_OWN_B   4
#define GB_OWN_I   8
#define GB_OWN_X   16
#define GB_OWN_ALL 31

GrB_Info GB_own_matrix      // GrB_SUCCESS or GrB_OUT_OF_MEMORY
(
    // input/output:
    GrB_Matrix A,           // matrix about to be modified in place: any of
                            // the selected A->[phbix] that is shared with
                            // another matrix is replaced with a private copy.
                            // Borrowed (shallow) arrays are left as they are.
    // input:
    int which,              // any combination of GB_OWN_P, ..., GB_OWN_X
    int nthreads            // # of threads to use for the copies
) ;

void GB_array_realloc       // reallocate an owned block (copied if shared)
(
    // input/output:
    void **p, uint64_t *mem, bool *shallow,     // slot to reallocate
    // input:
    uint64_t nitems_new,    // new # of items
    uint64_t size_of_item,  // size of each item
    int arena,              // arena, if the slot is absent on input
    // output:
    bool *ok                // true if successful, false otherwise
) ;

void GB_array_load          // load a block that came from the user
(
    // input/output:
    void **p, uint64_t *mem, bool *shallow,     // slot to set; absent on input
    // input:
    void *block,            // user block, or NULL
    uint64_t block_mem,     // its memsize and arena, GB_mem encoding
    bool readonly,          // if true, borrow; else take ownership
    bool add_to_memtable,   // if true, add to the debug memtable
    GB_array_hdr *spare,    // header to reuse, or NULL; always consumed
    // output:
    bool *ok                // true if successful, false if out of memory
) ;

GrB_Info GB_array_unload    // GrB_SUCCESS or GrB_OUT_OF_MEMORY
(
    // input/output:
    void **p, uint64_t *mem, bool *shallow,     // slot to unload; absent after
    // output:
    void **block,           // the raw block, or NULL
    uint64_t *block_mem,    // its memsize and arena, GB_mem encoding
    bool *readonly,         // true if the block was borrowed
    // input:
    int nthreads,           // # of threads to use if a copy is needed
    bool to_user,           // true: the block leaves GraphBLAS (it is removed
                            // from the debug memtable); false: it becomes
                            // internal workspace still freed by GB_FREE_MEMORY
    // output:
    GB_array_hdr **spare    // if non-NULL, the detached header is returned
                            // here (or NULL) instead of being freed
) ;

GB_array_hdr *GB_array_spare_take   // the spare header of a slot, or NULL
(
    // input/output:
    void **p, uint64_t *mem, bool *shallow      // slot; absent on output if a
                                                // header was taken
) ;

#endif
