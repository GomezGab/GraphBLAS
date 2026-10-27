//------------------------------------------------------------------------------
// GB_mex_init: initialize GraphBLAS
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Returns the status of all global settings.

#include "GB_mex.h"

#define USAGE "[nthreads format hyper_switch " \
"name version date about license compiledate compiletime api api_about" \
" chunk bitmap_switch] = GB_mex_init"

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    // Octave allocates only max (nargout, 1) output slots, so the outputs are
    // built in a local array and only the requested ones are returned
    mxArray *out [16] ;
    memset (out, 0, sizeof (out)) ;

//  mexPrintf ("usage:\n%s\n", USAGE) ;

    // finalize GraphBLAS but tell it that it can be called again
    GB_mx_at_exit ( ) ;

    // initialize GraphBLAS
//  GxB_init (GrB_NONBLOCKING, mxMalloc, NULL, NULL, mxFree) ;
    GB_mx_init ( ) ;

    // abort
    GB_Global_abort_set (GB_mx_abort) ;

    // enable debug malloc tracking
    GB_Global_malloc_tracking_set (true) ;

    // built-in default is by column
    GxB_Global_Option_set_(GxB_FORMAT, GxB_BY_COL) ;

    int nthreads ;
    GxB_Global_Option_get_(GxB_NTHREADS, &nthreads) ;
    out [0] = mxCreateDoubleScalar (nthreads) ;

    int format ;
    GxB_Global_Option_get_(GxB_FORMAT, &format) ;
    out [1] = mxCreateDoubleScalar (format) ;

    double hyper_switch ;
    GxB_Global_Option_get_(GxB_HYPER_SWITCH, &hyper_switch) ;
    out [2] = mxCreateDoubleScalar (hyper_switch) ;

    char *name ;
    GxB_Global_Option_get_(GxB_LIBRARY_NAME, &name) ;
    out [3] = mxCreateString (name) ;

    int version [3] ;
    GxB_Global_Option_get_(GxB_LIBRARY_VERSION, version) ;
    out [4] = mxCreateDoubleMatrix (1, 3, mxREAL) ;
    double *p = mxGetPr (out [4]) ;
    p [0] = version [0] ;
    p [1] = version [1] ;
    p [2] = version [2] ;

    char *date ;
    GxB_Global_Option_get_(GxB_LIBRARY_DATE, &date) ;
    out [5] = mxCreateString (date) ;

    char *about ;
    GxB_Global_Option_get_(GxB_LIBRARY_ABOUT, &about) ;
    out [6] = mxCreateString (about) ;

    char *license ;
    GxB_Global_Option_get_(GxB_LIBRARY_LICENSE, &license) ;
    out [7] = mxCreateString (license) ;

    char *compile_date ;
    GxB_Global_Option_get_(GxB_LIBRARY_COMPILE_DATE, &compile_date) ;
    out [8] = mxCreateString (compile_date) ;

    char *compile_time ;
    GxB_Global_Option_get_(GxB_LIBRARY_COMPILE_TIME, &compile_time) ;
    out [9] = mxCreateString (compile_time) ;

    int api [3] ;
    GxB_Global_Option_get_(GxB_API_VERSION, api) ;
    out [10] = mxCreateDoubleMatrix (1, 3, mxREAL) ;
    double *a = mxGetPr (out [10]) ;
    a [0] = api [0] ;
    a [1] = api [1] ;
    a [2] = api [2] ;

    char *api_about ;
    GxB_Global_Option_get_(GxB_API_ABOUT, &api_about) ;
    out [11] = mxCreateString (api_about) ;

    double chunk ;
    GxB_Global_Option_get_(GxB_CHUNK, &chunk) ;
    out [12] = mxCreateDoubleScalar (chunk) ;

    double bitmap_switch [GxB_NBITMAP_SWITCH] ;
    GxB_Global_Option_get_(GxB_BITMAP_SWITCH, bitmap_switch) ;
    out [13] = mxCreateDoubleMatrix (1, GxB_NBITMAP_SWITCH, mxREAL) ;
    double *bswitch = mxGetPr (out [13]) ;
    for (int k = 0 ; k < GxB_NBITMAP_SWITCH ; k++)
    {
        bswitch [k] = bitmap_switch [k] ;
    }

    // finalize GraphBLAS but tell it that it can be called again
    GB_mx_at_exit ( ) ;

    for (int k = 0 ; k < nargout && k < 16 ; k++)
    {
        pargout [k] = out [k] ;
    }
}
