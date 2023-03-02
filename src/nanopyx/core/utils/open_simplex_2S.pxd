
cdef extern from "OpenSimplex2S.h":

    ctypedef struct LatticePoint2D:
        int xsv, ysv
        double dx, dy

    ctypedef struct LatticePoint3D:
        double dxr, dyr, dzr
        int xrv, yrv, zrv
        struct _LatticePoint3D *nextOnFailure
        struct _LatticePoint3D *nextOnSuccess

    ctypedef struct LatticePoint4D:
    int xsv, ysv, zsv, wsv
    double dx, dy, dz, dw
    double xsi, ysi, zsi, wsi
    double ssiDelta

    ctypedef struct Grad2:
        double dx, dy

    ctypedef struct Grad3:
        double dx, dy, dz

    ctypedef struct Grad4:
        double dx, dy, dz, dw

    ctypedef struct OpenSimplexGradients:
        short *perm
        Grad2 *permGrad2
        Grad3 *permGrad3
        Grad4 *permGrad4

    ctypedef struct OpenSimplexEnv:
        Grad2 *GRADIENTS_2D
        Grad3 *GRADIENTS_3D
        Grad4 *GRADIENTS_4D
        LatticePoint2D **LOOKUP_2D
        LatticePoint3D **LOOKUP_3D
        LatticePoint4D **VERTICES_4D

    OpenSimplexEnv *initOpenSimplex()
    OpenSimplexGradients *newOpenSimplexGradients(OpenSimplexEnv *ose, long seed)
    double noise2(OpenSimplexEnv *ose, OpenSimplexGradients *osg, double x, double y)
    double noise2_XBeforeY(OpenSimplexEnv *ose, OpenSimplexGradients *osg, double x, double y)
    double noise3_Classic(OpenSimplexEnv *ose, OpenSimplexGradients *osg, double x, double y, double z)
    double noise3_XYBeforeZ(OpenSimplexEnv *ose, OpenSimplexGradients *osg, double x, double y, double z)
    double noise3_XZBeforeY(OpenSimplexEnv *ose, OpenSimplexGradients *osg, double x, double y, double z)
    double noise4_Classic(OpenSimplexEnv *ose, OpenSimplexGradients *osg, double x, double y, double z, double w)
    double noise4_XYBeforeZW(OpenSimplexEnv *ose, OpenSimplexGradients *osg, double x, double y, double z, double w)
    double noise4_XZBeforeYW(OpenSimplexEnv *ose, OpenSimplexGradients *osg, double x, double y, double z, double w)
    double noise4_XYZBeforeW(OpenSimplexEnv *ose, OpenSimplexGradients *osg, double x, double y, double z, double w)
