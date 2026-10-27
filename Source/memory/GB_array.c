//------------------------------------------------------------------------------
// GB_array.c: non-inline methods for reference-counted matrix arrays
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// See memory/include/GB_array.h for the slot states and the inline methods.

#include "GB.h"

//------------------------------------------------------------------------------
// GB_array_hdr_new: allocate a header for a block
//------------------------------------------------------------------------------

GB_array_hdr *GB_array_hdr_new      // the new header, or NULL if failed
(
    uint64_t block_mem      // memsize and arena of the block
)
{
    // the header lives in the same arena as the block
    uint64_t hdr_mem = GB_mem (GB_arena (block_mem), 0) ;
    GB_array_hdr *hdr = GB_MALLOC_MEMORY (1, sizeof (GB_array_hdr), &hdr_mem) ;
    if (hdr == NULL)
    {
        // out of memory
        return (NULL) ;
    }
    hdr->mem = block_mem ;
    hdr->hdr_mem = hdr_mem ;
    hdr->refs = 1 ;
    return (hdr) ;
}

//------------------------------------------------------------------------------
// GB_array_adopt: take ownership of a block that came from GB_malloc_memory
//------------------------------------------------------------------------------

// Used when workspace allocated by GB_malloc_memory becomes the content of a
// matrix (C->p = Cp ; C->p_mem = Cp_mem).  On failure the block is not freed;
// the caller still owns it and must free it.

GrB_Info GB_array_adopt     // GrB_SUCCESS, or GrB_OUT_OF_MEMORY
(
    // input/output:
    void **p, uint64_t *mem, bool *shallow,     // slot to set; absent on input
    // input:
    void *block,            // block from GB_malloc_memory, or NULL
    uint64_t block_mem      // its memsize and arena, from GB_malloc_memory
)
{
    ASSERT ((*p) == NULL) ;
    GB_array_spare_free (p, mem, shallow) ;
    if (block == NULL)
    {
        // nothing to adopt
        GB_array_clear (p, mem, shallow) ;
        return (GrB_SUCCESS) ;
    }
    GB_array_hdr *hdr = GB_array_hdr_new (block_mem) ;
    if (hdr == NULL)
    {
        // out of memory; the caller still owns the block
        GB_array_clear (p, mem, shallow) ;
        return (GrB_OUT_OF_MEMORY) ;
    }
    (*p) = block ;
    (*mem) = (uint64_t) (uintptr_t) hdr ;
    (*shallow) = false ;
    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// GB_array_spare_take: take a header from a slot for reuse by GB_array_load
//------------------------------------------------------------------------------

// If the slot holds a spare header, it is returned.  If the slot holds a
// block with a single owner, the block is freed (the caller is about to
// discard the slot's content anyway) and its header is returned.  Otherwise
// NULL is returned and the slot is unchanged.

GB_array_hdr *GB_array_spare_take   // the spare header of a slot, or NULL
(
    // input/output:
    void **p, uint64_t *mem, bool *shallow
)
{
    GB_array_hdr *spare = GB_array_spare_get (*p, *mem, *shallow) ;
    if (spare != NULL)
    {
        (*mem) = 0 ;
        return (spare) ;
    }
    GB_array_hdr *hdr = GB_array_hdr_get (*p, *mem, *shallow) ;
    if (hdr != NULL && GB_array_refs (*p, *mem, *shallow) == 1)
    {
        // the slot owns its block exclusively: free the block, keep the header
        uint64_t block_mem = hdr->mem ;
        GB_FREE_MEMORY (p, block_mem) ;
        hdr->mem = 0 ;
        hdr->refs = 0 ;
        GB_array_clear (p, mem, shallow) ;
        return (hdr) ;
    }
    return (NULL) ;
}

//------------------------------------------------------------------------------
// GB_array_malloc / GB_array_calloc / GB_array_xalloc
//------------------------------------------------------------------------------

// Allocate a block and its header.  If either allocation fails, nothing is
// allocated on output, the slot is absent, and NULL is returned.  Replaces:
//
//      A->c_mem = GB_mem (arena, 0) ;
//      A->c = GB_MALLOC_MEMORY (n, size, &(A->c_mem)) ;

void *GB_array_malloc       // pointer to the new block, or NULL if failed
(
    // input/output:
    void **p, uint64_t *mem, bool *shallow,     // slot to set; absent on input
    // input:
    uint64_t nitems,        // # of items to allocate
    uint64_t size_of_item,  // size of each item
    int arena               // arena to allocate the block and header in
)
{
    ASSERT ((*p) == NULL) ;
    uint64_t block_mem = GB_mem (arena, 0) ;
    void *block = GB_MALLOC_MEMORY (nitems, size_of_item, &block_mem) ;
    if (GB_array_adopt (p, mem, shallow, block, block_mem) != GrB_SUCCESS)
    {
        // out of memory
        GB_FREE_MEMORY (&block, block_mem) ;
        return (NULL) ;
    }
    return (block) ;
}

void *GB_array_calloc       // pointer to the new block, or NULL if failed
(
    // input/output:
    void **p, uint64_t *mem, bool *shallow,     // slot to set; absent on input
    // input:
    uint64_t nitems,        // # of items to allocate
    uint64_t size_of_item,  // size of each item
    int arena               // arena to allocate the block and header in
)
{
    ASSERT ((*p) == NULL) ;
    uint64_t block_mem = GB_mem (arena, 0) ;
    void *block = GB_CALLOC_MEMORY (nitems, size_of_item, &block_mem) ;
    if (GB_array_adopt (p, mem, shallow, block, block_mem) != GrB_SUCCESS)
    {
        // out of memory
        GB_FREE_MEMORY (&block, block_mem) ;
        return (NULL) ;
    }
    return (block) ;
}

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
)
{
    ASSERT ((*p) == NULL) ;
    uint64_t block_mem = GB_mem (arena, 0) ;
    void *block = GB_XALLOC_MEMORY (use_calloc, iso, nentries, sizeof_entry,
        &block_mem) ;
    if (GB_array_adopt (p, mem, shallow, block, block_mem) != GrB_SUCCESS)
    {
        // out of memory
        GB_FREE_MEMORY (&block, block_mem) ;
        return (NULL) ;
    }
    return (block) ;
}

//------------------------------------------------------------------------------
// GB_array_own: ensure a slot holds a block that may be written
//------------------------------------------------------------------------------

// On output the slot is absent (if it was absent on input) or owned with
// refs == 1.  If the slot was borrowed, or owned but shared, the first nbytes
// of the block are copied into a new block of nbytes bytes, and the old
// reference is dropped only after the copy is complete.  The raw pointer in
// the slot changes in that case, so callers must re-read it.
//
// This must be called before any in-place write to an array of a matrix that
// may be the result of GrB_dup, or that may be a shallow view.

GrB_Info GB_array_own       // GrB_SUCCESS or GrB_OUT_OF_MEMORY
(
    // input/output:
    void **p, uint64_t *mem, bool *shallow,     // slot to make writable
    // input:
    uint64_t nbytes,        // # of bytes of the block that are live
    int arena,              // arena for the copy, if one is needed
    int nthreads            // # of threads to use for the copy
)
{

    //--------------------------------------------------------------------------
    // quick return if the slot is absent or already exclusively owned
    //--------------------------------------------------------------------------

    void *old_block = (*p) ;
    if (old_block == NULL)
    {
        // absent: nothing to own
        return (GrB_SUCCESS) ;
    }

    if (!(*shallow))
    {
        int64_t refs ;
        GB_array_hdr *hdr = (GB_array_hdr *) (uintptr_t) (*mem) ;
        GB_ATOMIC_LOAD64 (refs, hdr->refs) ;
        ASSERT (refs >= 1) ;
        if (refs == 1)
        {
            // owned and not shared: already writable.  No other thread can
            // increment refs now, since a share requires a reference and
            // this slot holds the only one.
            return (GrB_SUCCESS) ;
        }
        // owned but shared: fall through and copy
    }

    //--------------------------------------------------------------------------
    // copy the live bytes into a new, exclusively owned block
    //--------------------------------------------------------------------------

    void *new_block = NULL ;
    uint64_t new_mem = 0 ;
    bool new_shallow = false ;
    if (GB_array_malloc (&new_block, &new_mem, &new_shallow, nbytes,
        sizeof (GB_void), arena) == NULL)
    {
        // out of memory; the slot is unchanged
        return (GrB_OUT_OF_MEMORY) ;
    }
    ASSERT (nbytes <= GB_array_memsize (old_block, *mem, *shallow)) ;
    GB_memcpy (new_block, old_block, nbytes, nthreads) ;

    //--------------------------------------------------------------------------
    // drop the old reference, after the copy, and install the new block
    //--------------------------------------------------------------------------

    GB_array_release (p, mem, shallow) ;
    GB_array_move (p, mem, shallow, &new_block, &new_mem, &new_shallow) ;
    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// GB_array_own_discard: make a slot exclusively owned, discarding its content
//------------------------------------------------------------------------------

// For a caller that overwrites the whole block: a shared block is replaced by
// a new uninitialized block of the same size in the same arena, and the old
// reference is dropped.  Nothing is copied.  Absent, borrowed, and exclusively
// owned slots are left as they are.

GrB_Info GB_array_own_discard   // GrB_SUCCESS or GrB_OUT_OF_MEMORY
(
    // input/output:
    void **p, uint64_t *mem, bool *shallow      // slot to make writable
)
{
    if ((*p) == NULL || (*shallow) || !GB_array_is_shared (*p, *mem, *shallow))
    { 
        // absent, borrowed, or already exclusively owned
        return (GrB_SUCCESS) ;
    }
    uint64_t nbytes = GB_array_memsize (*p, *mem, *shallow) ;
    int arena = GB_array_arena (*p, *mem, *shallow) ;
    GB_VSLOT_DECLARE (void *, new_block) ;
    if (GB_array_malloc (GB_VSLOT (new_block), nbytes, sizeof (GB_void), arena)
        == NULL)
    { 
        // out of memory; the slot is unchanged
        return (GrB_OUT_OF_MEMORY) ;
    }
    GB_array_release (p, mem, shallow) ;
    GB_array_move (p, mem, shallow, GB_VSLOT (new_block)) ;
    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// GB_own_matrix: make every shared array of a matrix private
//------------------------------------------------------------------------------

// Called before a matrix is modified in place (setElement, assign, in-place
// apply or mxm, unjumble, ...).  Each array that is shared with another
// matrix (after GrB_dup, or an internal share) is replaced with a private
// copy, so the other owners never see the modification.  Arrays with a single
// owner are left as they are, so this is O(1) in the common case.  Borrowed
// (shallow) arrays are also left as they are, to preserve the existing
// shallow semantics.
//
// Only the live part of each array is copied, as GB_dup_worker copied it
// before arrays were shared: the block may have slack beyond the live part
// (from GB_ix_realloc growth, or an array packed with a larger size), and
// that slack is not carried into the copy.  The copy can be grown later by
// GB_array_realloc.  The counts of A must be current: A->plen for p and h,
// A->nvals (entries held, zombies included) for i, b, and x.

static uint64_t GB_own_live_bytes   // # of live bytes of one array of A
(
    GrB_Matrix A,
    int slot                // one of GB_OWN_P, ..., GB_OWN_X
)
{
    // # of entries held: zombies count, as do all positions of a bitmap.
    // Not GB_nnz_held, which returns 0 when A->x is NULL.
    int64_t nheld = (A->p != NULL) ? A->nvals : GB_nnz_full (A) ;
    switch (slot)
    {
        case GB_OWN_P :
            // A->plen + 1 entries; A->plen == A->vdim if A is sparse
            return ((A->plen + 1) * (A->p_is_32 ? sizeof (uint32_t) :
                sizeof (uint64_t))) ;
        case GB_OWN_H :
            return (A->plen * (A->j_is_32 ? sizeof (uint32_t) :
                sizeof (uint64_t))) ;
        case GB_OWN_B :
            return (nheld) ;
        case GB_OWN_I :
            return (nheld * (A->i_is_32 ? sizeof (uint32_t) :
                sizeof (uint64_t))) ;
        default :
            return ((A->iso ? 1 : nheld) * A->type->size) ;
    }
}

GrB_Info GB_own_matrix      // GrB_SUCCESS or GrB_OUT_OF_MEMORY
(
    // input/output:
    GrB_Matrix A,           // matrix about to be modified in place
    // input:
    int which,              // any combination of GB_OWN_P, ..., GB_OWN_X
    int nthreads            // # of threads to use for the copies
)
{
    if (A == NULL)
    {
        return (GrB_SUCCESS) ;
    }
    int arena = A->data_arena ;
    GrB_Info info ;
    #define GB_OWN_SLOT(c,flag)                                             \
    if ((which & flag) && GB_array_is_shared (GB_CSLOT (A, c)))             \
    {                                                                       \
        info = GB_array_own (GB_SLOT (A, c), GB_own_live_bytes (A, flag),   \
            arena, nthreads) ;                                              \
        if (info != GrB_SUCCESS) return (info) ;                            \
    }
    GB_OWN_SLOT (p, GB_OWN_P) ;
    GB_OWN_SLOT (h, GB_OWN_H) ;
    GB_OWN_SLOT (b, GB_OWN_B) ;
    GB_OWN_SLOT (i, GB_OWN_I) ;
    GB_OWN_SLOT (x, GB_OWN_X) ;
    #undef GB_OWN_SLOT
    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// GB_own_discard: make every shared array private, discarding its content
//------------------------------------------------------------------------------

GrB_Info GB_own_discard     // GrB_SUCCESS or GrB_OUT_OF_MEMORY
(
    // input/output:
    GrB_Matrix A,           // matrix about to be overwritten in place
    // input:
    int which               // any combination of GB_OWN_P, ..., GB_OWN_X
)
{
    if (A == NULL)
    {
        return (GrB_SUCCESS) ;
    }
    GrB_Info info ;
    #define GB_DISCARD_SLOT(c,flag)                                         \
    if (which & flag)                                                       \
    {                                                                       \
        info = GB_array_own_discard (GB_SLOT (A, c)) ;                      \
        if (info != GrB_SUCCESS) return (info) ;                            \
    }
    GB_DISCARD_SLOT (p, GB_OWN_P) ;
    GB_DISCARD_SLOT (h, GB_OWN_H) ;
    GB_DISCARD_SLOT (b, GB_OWN_B) ;
    GB_DISCARD_SLOT (i, GB_OWN_I) ;
    GB_DISCARD_SLOT (x, GB_OWN_X) ;
    #undef GB_DISCARD_SLOT
    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// GB_own_snapshot_take / GB_own_snapshot_check: catch writes to shared arrays
//------------------------------------------------------------------------------

// A method that modifies C in place names the arrays it writes when it calls
// GB_own_matrix.  In a GB_DEBUG build, the shared arrays it did not name are
// hashed before the method runs and checked afterwards: if the block is still
// in the slot and its live bytes changed, the method wrote into an array it
// does not own.  A block that was replaced (owned, reallocated, or rebuilt by
// the method) is not checked.

#ifdef GB_DEBUG

static uint64_t GB_own_hash (const void *block, uint64_t nbytes)
{
    // FNV-1a
    const uint8_t *b = (const uint8_t *) block ;
    uint64_t h = 0xcbf29ce484222325ULL ;
    for (uint64_t k = 0 ; k < nbytes ; k++)
    {
        h ^= (uint64_t) b [k] ;
        h *= 0x100000001b3ULL ;
    }
    return (h) ;
}

void GB_own_snapshot_take   // record the shared arrays not in which
(
    GB_own_snapshot *snap,
    GrB_Matrix A,
    int which               // the arrays claimed by GB_own_matrix
)
{
    memset (snap, 0, sizeof (GB_own_snapshot)) ;
    if (A == NULL)
    {
        return ;
    }
    #define GB_SNAP(k,c,flag)                                               \
    if (!(which & flag) && GB_array_is_shared (GB_CSLOT (A, c)))            \
    {                                                                       \
        uint64_t n = GB_own_live_bytes (A, flag) ;                          \
        ASSERT (n <= GB_array_memsize (GB_CSLOT (A, c))) ;                  \
        snap->block [k] = A->c ;                                            \
        snap->mem [k] = A->c ## _mem ;                                      \
        snap->nbytes [k] = n ;                                              \
        snap->hash [k] = GB_own_hash (A->c, n) ;                            \
    }
    GB_SNAP (0, p, GB_OWN_P) ;
    GB_SNAP (1, h, GB_OWN_H) ;
    GB_SNAP (2, b, GB_OWN_B) ;
    GB_SNAP (3, i, GB_OWN_I) ;
    GB_SNAP (4, x, GB_OWN_X) ;
    #undef GB_SNAP
}

void GB_own_snapshot_check  // assert that no recorded array was written
(
    const GB_own_snapshot *snap,
    GrB_Matrix A
)
{
    if (A == NULL)
    {
        return ;
    }
    #define GB_SNAPCHECK(k,c)                                               \
    if (snap->block [k] != NULL && snap->block [k] == A->c                  \
        && snap->mem [k] == A->c ## _mem)                                   \
    {                                                                       \
        /* the same block is still in the slot: it must be unchanged */     \
        ASSERT (GB_array_memsize (GB_CSLOT (A, c)) >= snap->nbytes [k]) ;   \
        ASSERT (snap->hash [k] == GB_own_hash (A->c, snap->nbytes [k])) ;   \
    }
    GB_SNAPCHECK (0, p) ;
    GB_SNAPCHECK (1, h) ;
    GB_SNAPCHECK (2, b) ;
    GB_SNAPCHECK (3, i) ;
    GB_SNAPCHECK (4, x) ;
    #undef GB_SNAPCHECK
}

#endif

//------------------------------------------------------------------------------
// GB_array_realloc: reallocate an owned block (copied first if shared)
//------------------------------------------------------------------------------

// Replaces GB_REALLOC_MEMORY (A->c, n, size, &(A->c_mem), &ok).  The slot
// must be absent, or owned with refs == 1 (call GB_array_own first if it may
// be shared).  As with GB_realloc_memory, if the reallocation fails the
// original block is left intact and ok is false.

void GB_array_realloc       // reallocate an owned, unshared block
(
    // input/output:
    void **p, uint64_t *mem, bool *shallow,     // slot to reallocate
    // input:
    uint64_t nitems_new,    // new # of items
    uint64_t size_of_item,  // size of each item
    int arena,              // arena, if the slot is absent on input
    // output:
    bool *ok                // true if successful, false otherwise
)
{

    if ((*p) == NULL)
    {
        // absent: allocate a new block
        (*ok) = (GB_array_malloc (p, mem, shallow, nitems_new, size_of_item,
            arena) != NULL) ;
        return ;
    }

    ASSERT (!(*shallow)) ;
    if (GB_array_is_shared (*p, *mem, *shallow))
    {
        // the block is shared with another matrix: instead of copying it and
        // then reallocating the copy, allocate a block of the new size and
        // copy as much of the old block as fits.  The old block is untouched.
        void *new_block = NULL ;
        uint64_t new_mem = 0 ;
        bool new_shallow = false ;
        if (GB_array_malloc (&new_block, &new_mem, &new_shallow, nitems_new,
            size_of_item, GB_array_arena (*p, *mem, *shallow)) == NULL)
        {
            // out of memory; the slot is unchanged
            (*ok) = false ;
            return ;
        }
        uint64_t old_size = GB_array_memsize (*p, *mem, *shallow) ;
        uint64_t new_size = GB_array_memsize (new_block, new_mem, new_shallow) ;
        GB_memcpy (new_block, *p, GB_IMIN (old_size, new_size),
            GB_Context_nthreads_max ( )) ;
        GB_array_release (p, mem, shallow) ;
        GB_array_move (p, mem, shallow, &new_block, &new_mem, &new_shallow) ;
        (*ok) = true ;
        return ;
    }
    GB_array_hdr *hdr = (GB_array_hdr *) (uintptr_t) (*mem) ;
    ASSERT (GB_array_refs (*p, *mem, *shallow) == 1) ;

    // reallocate the block; hdr->mem is updated in place on success and left
    // unchanged on failure, matching GB_realloc_memory
    uint64_t block_mem = hdr->mem ;
    void *block = GB_realloc_memory (nitems_new, size_of_item, *p, &block_mem,
        ok) ;
    if (*ok)
    {
        (*p) = block ;
        hdr->mem = block_mem ;
    }
}

//------------------------------------------------------------------------------
// GB_array_load: load a block that came from the user application
//------------------------------------------------------------------------------

// Used by GxB_Vector_load, GxB_load_*_from_Container, GxB_*_pack, and
// GrB_*_import.  If readonly is true the block is borrowed and remains the
// property of the user application.  Otherwise, GraphBLAS takes ownership;
// the block must have been allocated by the malloc function of the arena
// encoded in block_mem.  If the header cannot be allocated, ok is false and
// the block remains the property of the caller.

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
)
{
    ASSERT ((*p) == NULL) ;
    GB_array_spare_free (p, mem, shallow) ;
    (*ok) = true ;
    if (block == NULL || readonly)
    {
        // absent or borrowed: no header is needed
        if (spare != NULL)
        {
            uint64_t hdr_mem = spare->hdr_mem ;
            GB_FREE_MEMORY (&spare, hdr_mem) ;
        }
        if (block == NULL)
        {
            GB_array_clear (p, mem, shallow) ;
        }
        else
        {
            GB_array_borrow (p, mem, shallow, block, block_mem) ;
        }
    }
    else if (spare != NULL)
    {
        // take ownership using the spare header: this cannot fail
        spare->mem = block_mem ;
        spare->refs = 1 ;
        (*p) = block ;
        (*mem) = (uint64_t) (uintptr_t) spare ;
        (*shallow) = false ;
        if (add_to_memtable)
        {
            GB_Global_memtable_add (block, block_mem) ;
        }
    }
    else
    {
        (*ok) = (GB_array_adopt (p, mem, shallow, block, block_mem)
            == GrB_SUCCESS) ;
        if (*ok && add_to_memtable)
        {
            GB_Global_memtable_add (block, block_mem) ;
        }
    }
}

//------------------------------------------------------------------------------
// GB_array_unload: hand a block to the user application
//------------------------------------------------------------------------------

// Used by GxB_Vector_unload, GxB_unload_*_into_Container, GxB_*_unpack, and
// GxB_*_export.  The slot is absent on output.  A borrowed block is returned
// as-is with readonly true.  An owned block that is shared with another
// matrix is first copied in full (all memsize bytes, since the user receives
// the memsize), so that the other owners are unaffected.  The returned block
// is always a pointer from the malloc function of its arena, and the header
// is freed here, so the user application may free the block directly.

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
)
{

    (*block) = NULL ;
    (*block_mem) = 0 ;
    (*readonly) = false ;
    if (spare != NULL)
    {
        (*spare) = NULL ;
    }

    if ((*p) == NULL)
    {
        // absent: nothing to unload (a spare header, if any, is freed)
        GB_array_release (p, mem, shallow) ;
        return (GrB_SUCCESS) ;
    }

    if (*shallow)
    {
        // borrowed: return the pointer and forget it
        (*block) = (*p) ;
        (*block_mem) = (*mem) ;
        (*readonly) = true ;
        GB_array_clear (p, mem, shallow) ;
        return (GrB_SUCCESS) ;
    }

    // owned: make sure this slot is the only owner
    GB_array_hdr *hdr = (GB_array_hdr *) (uintptr_t) (*mem) ;
    GrB_Info info = GB_array_own (p, mem, shallow, GB_memsize (hdr->mem),
        GB_arena (hdr->mem), nthreads) ;
    if (info != GrB_SUCCESS)
    {
        // out of memory; the slot is unchanged
        return (info) ;
    }

    // detach the block from its header, and free the header
    hdr = (GB_array_hdr *) (uintptr_t) (*mem) ;
    ASSERT (hdr->refs == 1) ;
    (*block) = (*p) ;
    (*block_mem) = hdr->mem ;
    if (to_user)
    {
        GB_Global_memtable_remove (*block) ;
    }
    if (spare != NULL)
    {
        // hand the header back to the caller for reuse
        hdr->mem = 0 ;
        hdr->refs = 0 ;
        (*spare) = hdr ;
    }
    else
    {
        uint64_t hdr_mem = hdr->hdr_mem ;
        GB_FREE_MEMORY (&hdr, hdr_mem) ;
    }
    GB_array_clear (p, mem, shallow) ;
    return (GrB_SUCCESS) ;
}
