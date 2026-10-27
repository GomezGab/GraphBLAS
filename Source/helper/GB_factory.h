//------------------------------------------------------------------------------
// GB_factory.h: for testing factory kernels
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_FACTORY_H
#define GB_FACTORY_H

// For testing in MATLAB/Octave, factory kernels can be disabled.  The flag
// is always defined (GB_helper.c) so the test interface can link to it from
// either library build; it only takes effect in a GBMATLAB build.
extern bool GB_factory_kernels_enabled ;
#ifdef GBMATLAB
#define GB_IF_FACTORY_KERNELS_ENABLED if (GB_factory_kernels_enabled)
#else
#define GB_IF_FACTORY_KERNELS_ENABLED
#endif

#endif

