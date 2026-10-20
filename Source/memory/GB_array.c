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

static GB_array_hdr *GB_array_hdr_new   // the new header, or NULL if failed
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

bool GB_array_adopt         // true if successful, false if out of memory
(
    // input/output:
    void **p, uint64_t *mem, bool *shallow,     // slot to set; absent on input
    // input:
    void *block,            // block from GB_malloc_memory, or NULL
    uint64_t block_mem      // its memsize and arena, from GB_malloc_memory
)
{
    ASSERT ((*p) == NULL) ;
    if (block == NULL)
    {
        // nothing to adopt
        GB_array_clear (p, mem, shallow) ;
        return (true) ;
    }
    GB_array_hdr *hdr = GB_array_hdr_new (block_mem) ;
    if (hdr == NULL)
    {
        // out of memory; the caller still owns the block
        GB_array_clear (p, mem, shallow) ;
        return (false) ;
    }
    (*p) = block ;
    (*mem) = (uint64_t) (uintptr_t) hdr ;
    (*shallow) = false ;
    return (true) ;
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
    if (!GB_array_adopt (p, mem, shallow, block, block_mem))
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
    if (!GB_array_adopt (p, mem, shallow, block, block_mem))
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
    if (!GB_array_adopt (p, mem, shallow, block, block_mem))
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
// GB_array_realloc: reallocate an owned, unshared block
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
    // output:
    bool *ok                // true if successful, false if out of memory
)
{
    ASSERT ((*p) == NULL) ;
    (*ok) = true ;
    if (block == NULL)
    {
        GB_array_clear (p, mem, shallow) ;
    }
    else if (readonly)
    {
        GB_array_borrow (p, mem, shallow, block, block_mem) ;
    }
    else
    {
        (*ok) = GB_array_adopt (p, mem, shallow, block, block_mem) ;
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
    int nthreads            // # of threads to use if a copy is needed
)
{

    (*block) = NULL ;
    (*block_mem) = 0 ;
    (*readonly) = false ;

    if ((*p) == NULL)
    {
        // absent: nothing to unload
        GB_array_clear (p, mem, shallow) ;
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
    uint64_t hdr_mem = hdr->hdr_mem ;
    GB_Global_memtable_remove (*block) ;
    GB_FREE_MEMORY (&hdr, hdr_mem) ;
    GB_array_clear (p, mem, shallow) ;
    return (GrB_SUCCESS) ;
}
