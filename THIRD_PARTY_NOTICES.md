# Third-party origins and notices

PyHREBSD is a Python implementation of HR-EBSD analysis. Some of its routines
were developed using OpenXY source code as a reference. In particular,
`pyhrebsd/correlation.py` identifies an OpenXY ROI-correlation port, while
`pyhrebsd/analysis.py` identifies OpenXY's `CalcFShift.m` as a basis for part
of the deformation analysis.
Similar scientific methods, parameter choices, or numerical results alone do
not establish that a particular source file is a copy of either project.

## OpenXY

OpenXY is copyright (c) 2009-2020 David Fullwood Research Group / Brigham
Young University. Its permissive three-clause BSD-style license allows use,
modification, and redistribution, subject to retention of its copyright
notice, license conditions, and disclaimer. The names of its authors and
contributors may not be used to endorse a PyHREBSD product without their
written permission. The complete license is in
[`third_party_licenses/OpenXY-LICENSE.txt`](third_party_licenses/OpenXY-LICENSE.txt).
The reference source is <https://github.com/KacherLab/OpenXY>.

This acknowledgment does not imply that OpenXY, its authors, or its
institutions endorse PyHREBSD. The license for original PyHREBSD code is in
[`LICENSE`](LICENSE) and is separate from the license listed here.
