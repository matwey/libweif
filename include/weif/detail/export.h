/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Copyright (C) 2022-2024  Matwey V. Kornilov <matwey.kornilov@gmail.com>
 */

#ifndef _WEIF_DETAIL_EXPORT_H
#define _WEIF_DETAIL_EXPORT_H

#include <weif_export.h>

/*
 * Class templates are header-only from the Windows DLL point of view:
 * __declspec(dllimport) cannot be applied to definitions that live in
 * headers, so on Windows the export attribute is moved to the explicit
 * instantiation definitions instead (WEIF_INSTANTIATION_EXPORT).
 *
 * On ELF/Mach-O the visibility attribute stays on the class template,
 * exactly as before, so the existing explicit-instantiation setup keeps
 * working unchanged.
 */
#if defined(_WIN32) || defined(__CYGWIN__)
#  define WEIF_TEMPLATE_EXPORT
#  define WEIF_INSTANTIATION_EXPORT WEIF_EXPORT
#else
#  define WEIF_TEMPLATE_EXPORT WEIF_EXPORT
#  define WEIF_INSTANTIATION_EXPORT
#endif

#endif // _WEIF_DETAIL_EXPORT_H
