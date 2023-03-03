#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>
#include "OpenSimplex2F.h"

#define PSIZE 2048
#define PMASK 2047
#define N2 0.01001634121365712
#define N3 0.030485933181293584
#define N4 0.009202377986303158

/*
 * Utility
 */

int _fastFloor(double x)
{
	int xi = (int)x;
	return x < xi ? xi - 1 : xi;
}

Grad2 *_newGrad2Arr(unsigned int size)
{
	return (Grad2 *)malloc(sizeof(Grad2) * size);
}

Grad3 *_newGrad3Arr(unsigned int size)
{
	return (Grad3 *)malloc(sizeof(Grad3) * size);
}

Grad4 *_newGrad4Arr(unsigned int size)
{
	return (Grad4 *)malloc(sizeof(Grad4) * size);
}

short *_newShortArr(unsigned int size)
{
	return (short *)malloc(sizeof(short) * size);
}

Grad2 _newGrad2(double dx, double dy)
{
	Grad2 grad2;
	grad2.dx = dx;
	grad2.dy = dy;
	return grad2;
}

Grad3 _newGrad3(double dx, double dy, double dz)
{
	Grad3 grad3;
	grad3.dx = dx;
	grad3.dy = dy;
	grad3.dz = dz;
	return grad3;
}

Grad4 _newGrad4(double dx, double dy, double dz, double dw)
{
	Grad4 grad4;
	grad4.dx = dx;
	grad4.dy = dy;
	grad4.dz = dz;
	grad4.dw = dw;
	return grad4;
}

Grad2 *_newGrad2ConstArray()
{
	Grad2 *arr = (Grad2 *)malloc(sizeof(Grad2) * 24);
	int i = 0;
	arr[i++] = _newGrad2(0.130526192220052, 0.99144486137381);
	arr[i++] = _newGrad2(0.38268343236509, 0.923879532511287);
	arr[i++] = _newGrad2(0.608761429008721, 0.793353340291235);
	arr[i++] = _newGrad2(0.793353340291235, 0.608761429008721);
	arr[i++] = _newGrad2(0.923879532511287, 0.38268343236509);
	arr[i++] = _newGrad2(0.99144486137381, 0.130526192220051);
	arr[i++] = _newGrad2(0.99144486137381, -0.130526192220051);
	arr[i++] = _newGrad2(0.923879532511287, -0.38268343236509);
	arr[i++] = _newGrad2(0.793353340291235, -0.60876142900872);
	arr[i++] = _newGrad2(0.608761429008721, -0.793353340291235);
	arr[i++] = _newGrad2(0.38268343236509, -0.923879532511287);
	arr[i++] = _newGrad2(0.130526192220052, -0.99144486137381);
	arr[i++] = _newGrad2(-0.130526192220052, -0.99144486137381);
	arr[i++] = _newGrad2(-0.38268343236509, -0.923879532511287);
	arr[i++] = _newGrad2(-0.608761429008721, -0.793353340291235);
	arr[i++] = _newGrad2(-0.793353340291235, -0.608761429008721);
	arr[i++] = _newGrad2(-0.923879532511287, -0.38268343236509);
	arr[i++] = _newGrad2(-0.99144486137381, -0.130526192220052);
	arr[i++] = _newGrad2(-0.99144486137381, 0.130526192220051);
	arr[i++] = _newGrad2(-0.923879532511287, 0.38268343236509);
	arr[i++] = _newGrad2(-0.793353340291235, 0.608761429008721);
	arr[i++] = _newGrad2(-0.608761429008721, 0.793353340291235);
	arr[i++] = _newGrad2(-0.38268343236509, 0.923879532511287);
	arr[i++] = _newGrad2(-0.130526192220052, 0.99144486137381);
	Grad2 *gradients2D = _newGrad2Arr(PSIZE);
	for (int i = 0; i < 24; i++)
	{
		arr[i].dx /= N2;
		arr[i].dy /= N2;
	}
	for (int i = 0; i < PSIZE; i++)
	{
		gradients2D[i] = arr[i % 24];
	}
	return gradients2D;
}

Grad3 *_newGrad3ConstArray()
{
	Grad3 *arr = (Grad3 *)malloc(sizeof(Grad3) * 48);
	int i = 0;
	arr[i++] = _newGrad3(-2.22474487139, -2.22474487139, -1.0);
	arr[i++] = _newGrad3(-2.22474487139, -2.22474487139, 1.0);
	arr[i++] = _newGrad3(-3.0862664687972017, -1.1721513422464978, 0.0);
	arr[i++] = _newGrad3(-1.1721513422464978, -3.0862664687972017, 0.0);
	arr[i++] = _newGrad3(-2.22474487139, -1.0, -2.22474487139);
	arr[i++] = _newGrad3(-2.22474487139, 1.0, -2.22474487139);
	arr[i++] = _newGrad3(-1.1721513422464978, 0.0, -3.0862664687972017);
	arr[i++] = _newGrad3(-3.0862664687972017, 0.0, -1.1721513422464978);
	arr[i++] = _newGrad3(-2.22474487139, -1.0, 2.22474487139);
	arr[i++] = _newGrad3(-2.22474487139, 1.0, 2.22474487139);
	arr[i++] = _newGrad3(-3.0862664687972017, 0.0, 1.1721513422464978);
	arr[i++] = _newGrad3(-1.1721513422464978, 0.0, 3.0862664687972017);
	arr[i++] = _newGrad3(-2.22474487139, 2.22474487139, -1.0);
	arr[i++] = _newGrad3(-2.22474487139, 2.22474487139, 1.0);
	arr[i++] = _newGrad3(-1.1721513422464978, 3.0862664687972017, 0.0);
	arr[i++] = _newGrad3(-3.0862664687972017, 1.1721513422464978, 0.0);
	arr[i++] = _newGrad3(-1.0, -2.22474487139, -2.22474487139);
	arr[i++] = _newGrad3(1.0, -2.22474487139, -2.22474487139);
	arr[i++] = _newGrad3(0.0, -3.0862664687972017, -1.1721513422464978);
	arr[i++] = _newGrad3(0.0, -1.1721513422464978, -3.0862664687972017);
	arr[i++] = _newGrad3(-1.0, -2.22474487139, 2.22474487139);
	arr[i++] = _newGrad3(1.0, -2.22474487139, 2.22474487139);
	arr[i++] = _newGrad3(0.0, -1.1721513422464978, 3.0862664687972017);
	arr[i++] = _newGrad3(0.0, -3.0862664687972017, 1.1721513422464978);
	arr[i++] = _newGrad3(-1.0, 2.22474487139, -2.22474487139);
	arr[i++] = _newGrad3(1.0, 2.22474487139, -2.22474487139);
	arr[i++] = _newGrad3(0.0, 1.1721513422464978, -3.0862664687972017);
	arr[i++] = _newGrad3(0.0, 3.0862664687972017, -1.1721513422464978);
	arr[i++] = _newGrad3(-1.0, 2.22474487139, 2.22474487139);
	arr[i++] = _newGrad3(1.0, 2.22474487139, 2.22474487139);
	arr[i++] = _newGrad3(0.0, 3.0862664687972017, 1.1721513422464978);
	arr[i++] = _newGrad3(0.0, 1.1721513422464978, 3.0862664687972017);
	arr[i++] = _newGrad3(2.22474487139, -2.22474487139, -1.0);
	arr[i++] = _newGrad3(2.22474487139, -2.22474487139, 1.0);
	arr[i++] = _newGrad3(1.1721513422464978, -3.0862664687972017, 0.0);
	arr[i++] = _newGrad3(3.0862664687972017, -1.1721513422464978, 0.0);
	arr[i++] = _newGrad3(2.22474487139, -1.0, -2.22474487139);
	arr[i++] = _newGrad3(2.22474487139, 1.0, -2.22474487139);
	arr[i++] = _newGrad3(3.0862664687972017, 0.0, -1.1721513422464978);
	arr[i++] = _newGrad3(1.1721513422464978, 0.0, -3.0862664687972017);
	arr[i++] = _newGrad3(2.22474487139, -1.0, 2.22474487139);
	arr[i++] = _newGrad3(2.22474487139, 1.0, 2.22474487139);
	arr[i++] = _newGrad3(1.1721513422464978, 0.0, 3.0862664687972017);
	arr[i++] = _newGrad3(3.0862664687972017, 0.0, 1.1721513422464978);
	arr[i++] = _newGrad3(2.22474487139, 2.22474487139, -1.0);
	arr[i++] = _newGrad3(2.22474487139, 2.22474487139, 1.0);
	arr[i++] = _newGrad3(3.0862664687972017, 1.1721513422464978, 0.0);
	arr[i++] = _newGrad3(1.1721513422464978, 3.0862664687972017, 0.0);
	Grad3 *gradients3D = _newGrad3Arr(PSIZE);
	for (int i = 0; i < 48; i++)
	{
		arr[i].dx /= N3;
		arr[i].dy /= N3;
		arr[i].dz /= N3;
	}
	for (int i = 0; i < PSIZE; i++)
	{
		gradients3D[i] = arr[i % 48];
	}
	return gradients3D;
}

Grad4 *_newGrad4ConstArray()
{
	Grad4 *arr = (Grad4 *)malloc(sizeof(Grad4) * 160);
	int i = 0;
	arr[i++] = _newGrad4(-0.753341017856078, -0.37968289875261624, -0.37968289875261624, -0.37968289875261624);
	arr[i++] = _newGrad4(-0.7821684431180708, -0.4321472685365301, -0.4321472685365301, 0.12128480194602098);
	arr[i++] = _newGrad4(-0.7821684431180708, -0.4321472685365301, 0.12128480194602098, -0.4321472685365301);
	arr[i++] = _newGrad4(-0.7821684431180708, 0.12128480194602098, -0.4321472685365301, -0.4321472685365301);
	arr[i++] = _newGrad4(-0.8586508742123365, -0.508629699630796, 0.044802370851755174, 0.044802370851755174);
	arr[i++] = _newGrad4(-0.8586508742123365, 0.044802370851755174, -0.508629699630796, 0.044802370851755174);
	arr[i++] = _newGrad4(-0.8586508742123365, 0.044802370851755174, 0.044802370851755174, -0.508629699630796);
	arr[i++] = _newGrad4(-0.9982828964265062, -0.03381941603233842, -0.03381941603233842, -0.03381941603233842);
	arr[i++] = _newGrad4(-0.37968289875261624, -0.753341017856078, -0.37968289875261624, -0.37968289875261624);
	arr[i++] = _newGrad4(-0.4321472685365301, -0.7821684431180708, -0.4321472685365301, 0.12128480194602098);
	arr[i++] = _newGrad4(-0.4321472685365301, -0.7821684431180708, 0.12128480194602098, -0.4321472685365301);
	arr[i++] = _newGrad4(0.12128480194602098, -0.7821684431180708, -0.4321472685365301, -0.4321472685365301);
	arr[i++] = _newGrad4(-0.508629699630796, -0.8586508742123365, 0.044802370851755174, 0.044802370851755174);
	arr[i++] = _newGrad4(0.044802370851755174, -0.8586508742123365, -0.508629699630796, 0.044802370851755174);
	arr[i++] = _newGrad4(0.044802370851755174, -0.8586508742123365, 0.044802370851755174, -0.508629699630796);
	arr[i++] = _newGrad4(-0.03381941603233842, -0.9982828964265062, -0.03381941603233842, -0.03381941603233842);
	arr[i++] = _newGrad4(-0.37968289875261624, -0.37968289875261624, -0.753341017856078, -0.37968289875261624);
	arr[i++] = _newGrad4(-0.4321472685365301, -0.4321472685365301, -0.7821684431180708, 0.12128480194602098);
	arr[i++] = _newGrad4(-0.4321472685365301, 0.12128480194602098, -0.7821684431180708, -0.4321472685365301);
	arr[i++] = _newGrad4(0.12128480194602098, -0.4321472685365301, -0.7821684431180708, -0.4321472685365301);
	arr[i++] = _newGrad4(-0.508629699630796, 0.044802370851755174, -0.8586508742123365, 0.044802370851755174);
	arr[i++] = _newGrad4(0.044802370851755174, -0.508629699630796, -0.8586508742123365, 0.044802370851755174);
	arr[i++] = _newGrad4(0.044802370851755174, 0.044802370851755174, -0.8586508742123365, -0.508629699630796);
	arr[i++] = _newGrad4(-0.03381941603233842, -0.03381941603233842, -0.9982828964265062, -0.03381941603233842);
	arr[i++] = _newGrad4(-0.37968289875261624, -0.37968289875261624, -0.37968289875261624, -0.753341017856078);
	arr[i++] = _newGrad4(-0.4321472685365301, -0.4321472685365301, 0.12128480194602098, -0.7821684431180708);
	arr[i++] = _newGrad4(-0.4321472685365301, 0.12128480194602098, -0.4321472685365301, -0.7821684431180708);
	arr[i++] = _newGrad4(0.12128480194602098, -0.4321472685365301, -0.4321472685365301, -0.7821684431180708);
	arr[i++] = _newGrad4(-0.508629699630796, 0.044802370851755174, 0.044802370851755174, -0.8586508742123365);
	arr[i++] = _newGrad4(0.044802370851755174, -0.508629699630796, 0.044802370851755174, -0.8586508742123365);
	arr[i++] = _newGrad4(0.044802370851755174, 0.044802370851755174, -0.508629699630796, -0.8586508742123365);
	arr[i++] = _newGrad4(-0.03381941603233842, -0.03381941603233842, -0.03381941603233842, -0.9982828964265062);
	arr[i++] = _newGrad4(-0.6740059517812944, -0.3239847771997537, -0.3239847771997537, 0.5794684678643381);
	arr[i++] = _newGrad4(-0.7504883828755602, -0.4004672082940195, 0.15296486218853164, 0.5029860367700724);
	arr[i++] = _newGrad4(-0.7504883828755602, 0.15296486218853164, -0.4004672082940195, 0.5029860367700724);
	arr[i++] = _newGrad4(-0.8828161875373585, 0.08164729285680945, 0.08164729285680945, 0.4553054119602712);
	arr[i++] = _newGrad4(-0.4553054119602712, -0.08164729285680945, -0.08164729285680945, 0.8828161875373585);
	arr[i++] = _newGrad4(-0.5029860367700724, -0.15296486218853164, 0.4004672082940195, 0.7504883828755602);
	arr[i++] = _newGrad4(-0.5029860367700724, 0.4004672082940195, -0.15296486218853164, 0.7504883828755602);
	arr[i++] = _newGrad4(-0.5794684678643381, 0.3239847771997537, 0.3239847771997537, 0.6740059517812944);
	arr[i++] = _newGrad4(-0.3239847771997537, -0.6740059517812944, -0.3239847771997537, 0.5794684678643381);
	arr[i++] = _newGrad4(-0.4004672082940195, -0.7504883828755602, 0.15296486218853164, 0.5029860367700724);
	arr[i++] = _newGrad4(0.15296486218853164, -0.7504883828755602, -0.4004672082940195, 0.5029860367700724);
	arr[i++] = _newGrad4(0.08164729285680945, -0.8828161875373585, 0.08164729285680945, 0.4553054119602712);
	arr[i++] = _newGrad4(-0.08164729285680945, -0.4553054119602712, -0.08164729285680945, 0.8828161875373585);
	arr[i++] = _newGrad4(-0.15296486218853164, -0.5029860367700724, 0.4004672082940195, 0.7504883828755602);
	arr[i++] = _newGrad4(0.4004672082940195, -0.5029860367700724, -0.15296486218853164, 0.7504883828755602);
	arr[i++] = _newGrad4(0.3239847771997537, -0.5794684678643381, 0.3239847771997537, 0.6740059517812944);
	arr[i++] = _newGrad4(-0.3239847771997537, -0.3239847771997537, -0.6740059517812944, 0.5794684678643381);
	arr[i++] = _newGrad4(-0.4004672082940195, 0.15296486218853164, -0.7504883828755602, 0.5029860367700724);
	arr[i++] = _newGrad4(0.15296486218853164, -0.4004672082940195, -0.7504883828755602, 0.5029860367700724);
	arr[i++] = _newGrad4(0.08164729285680945, 0.08164729285680945, -0.8828161875373585, 0.4553054119602712);
	arr[i++] = _newGrad4(-0.08164729285680945, -0.08164729285680945, -0.4553054119602712, 0.8828161875373585);
	arr[i++] = _newGrad4(-0.15296486218853164, 0.4004672082940195, -0.5029860367700724, 0.7504883828755602);
	arr[i++] = _newGrad4(0.4004672082940195, -0.15296486218853164, -0.5029860367700724, 0.7504883828755602);
	arr[i++] = _newGrad4(0.3239847771997537, 0.3239847771997537, -0.5794684678643381, 0.6740059517812944);
	arr[i++] = _newGrad4(-0.6740059517812944, -0.3239847771997537, 0.5794684678643381, -0.3239847771997537);
	arr[i++] = _newGrad4(-0.7504883828755602, -0.4004672082940195, 0.5029860367700724, 0.15296486218853164);
	arr[i++] = _newGrad4(-0.7504883828755602, 0.15296486218853164, 0.5029860367700724, -0.4004672082940195);
	arr[i++] = _newGrad4(-0.8828161875373585, 0.08164729285680945, 0.4553054119602712, 0.08164729285680945);
	arr[i++] = _newGrad4(-0.4553054119602712, -0.08164729285680945, 0.8828161875373585, -0.08164729285680945);
	arr[i++] = _newGrad4(-0.5029860367700724, -0.15296486218853164, 0.7504883828755602, 0.4004672082940195);
	arr[i++] = _newGrad4(-0.5029860367700724, 0.4004672082940195, 0.7504883828755602, -0.15296486218853164);
	arr[i++] = _newGrad4(-0.5794684678643381, 0.3239847771997537, 0.6740059517812944, 0.3239847771997537);
	arr[i++] = _newGrad4(-0.3239847771997537, -0.6740059517812944, 0.5794684678643381, -0.3239847771997537);
	arr[i++] = _newGrad4(-0.4004672082940195, -0.7504883828755602, 0.5029860367700724, 0.15296486218853164);
	arr[i++] = _newGrad4(0.15296486218853164, -0.7504883828755602, 0.5029860367700724, -0.4004672082940195);
	arr[i++] = _newGrad4(0.08164729285680945, -0.8828161875373585, 0.4553054119602712, 0.08164729285680945);
	arr[i++] = _newGrad4(-0.08164729285680945, -0.4553054119602712, 0.8828161875373585, -0.08164729285680945);
	arr[i++] = _newGrad4(-0.15296486218853164, -0.5029860367700724, 0.7504883828755602, 0.4004672082940195);
	arr[i++] = _newGrad4(0.4004672082940195, -0.5029860367700724, 0.7504883828755602, -0.15296486218853164);
	arr[i++] = _newGrad4(0.3239847771997537, -0.5794684678643381, 0.6740059517812944, 0.3239847771997537);
	arr[i++] = _newGrad4(-0.3239847771997537, -0.3239847771997537, 0.5794684678643381, -0.6740059517812944);
	arr[i++] = _newGrad4(-0.4004672082940195, 0.15296486218853164, 0.5029860367700724, -0.7504883828755602);
	arr[i++] = _newGrad4(0.15296486218853164, -0.4004672082940195, 0.5029860367700724, -0.7504883828755602);
	arr[i++] = _newGrad4(0.08164729285680945, 0.08164729285680945, 0.4553054119602712, -0.8828161875373585);
	arr[i++] = _newGrad4(-0.08164729285680945, -0.08164729285680945, 0.8828161875373585, -0.4553054119602712);
	arr[i++] = _newGrad4(-0.15296486218853164, 0.4004672082940195, 0.7504883828755602, -0.5029860367700724);
	arr[i++] = _newGrad4(0.4004672082940195, -0.15296486218853164, 0.7504883828755602, -0.5029860367700724);
	arr[i++] = _newGrad4(0.3239847771997537, 0.3239847771997537, 0.6740059517812944, -0.5794684678643381);
	arr[i++] = _newGrad4(-0.6740059517812944, 0.5794684678643381, -0.3239847771997537, -0.3239847771997537);
	arr[i++] = _newGrad4(-0.7504883828755602, 0.5029860367700724, -0.4004672082940195, 0.15296486218853164);
	arr[i++] = _newGrad4(-0.7504883828755602, 0.5029860367700724, 0.15296486218853164, -0.4004672082940195);
	arr[i++] = _newGrad4(-0.8828161875373585, 0.4553054119602712, 0.08164729285680945, 0.08164729285680945);
	arr[i++] = _newGrad4(-0.4553054119602712, 0.8828161875373585, -0.08164729285680945, -0.08164729285680945);
	arr[i++] = _newGrad4(-0.5029860367700724, 0.7504883828755602, -0.15296486218853164, 0.4004672082940195);
	arr[i++] = _newGrad4(-0.5029860367700724, 0.7504883828755602, 0.4004672082940195, -0.15296486218853164);
	arr[i++] = _newGrad4(-0.5794684678643381, 0.6740059517812944, 0.3239847771997537, 0.3239847771997537);
	arr[i++] = _newGrad4(-0.3239847771997537, 0.5794684678643381, -0.6740059517812944, -0.3239847771997537);
	arr[i++] = _newGrad4(-0.4004672082940195, 0.5029860367700724, -0.7504883828755602, 0.15296486218853164);
	arr[i++] = _newGrad4(0.15296486218853164, 0.5029860367700724, -0.7504883828755602, -0.4004672082940195);
	arr[i++] = _newGrad4(0.08164729285680945, 0.4553054119602712, -0.8828161875373585, 0.08164729285680945);
	arr[i++] = _newGrad4(-0.08164729285680945, 0.8828161875373585, -0.4553054119602712, -0.08164729285680945);
	arr[i++] = _newGrad4(-0.15296486218853164, 0.7504883828755602, -0.5029860367700724, 0.4004672082940195);
	arr[i++] = _newGrad4(0.4004672082940195, 0.7504883828755602, -0.5029860367700724, -0.15296486218853164);
	arr[i++] = _newGrad4(0.3239847771997537, 0.6740059517812944, -0.5794684678643381, 0.3239847771997537);
	arr[i++] = _newGrad4(-0.3239847771997537, 0.5794684678643381, -0.3239847771997537, -0.6740059517812944);
	arr[i++] = _newGrad4(-0.4004672082940195, 0.5029860367700724, 0.15296486218853164, -0.7504883828755602);
	arr[i++] = _newGrad4(0.15296486218853164, 0.5029860367700724, -0.4004672082940195, -0.7504883828755602);
	arr[i++] = _newGrad4(0.08164729285680945, 0.4553054119602712, 0.08164729285680945, -0.8828161875373585);
	arr[i++] = _newGrad4(-0.08164729285680945, 0.8828161875373585, -0.08164729285680945, -0.4553054119602712);
	arr[i++] = _newGrad4(-0.15296486218853164, 0.7504883828755602, 0.4004672082940195, -0.5029860367700724);
	arr[i++] = _newGrad4(0.4004672082940195, 0.7504883828755602, -0.15296486218853164, -0.5029860367700724);
	arr[i++] = _newGrad4(0.3239847771997537, 0.6740059517812944, 0.3239847771997537, -0.5794684678643381);
	arr[i++] = _newGrad4(0.5794684678643381, -0.6740059517812944, -0.3239847771997537, -0.3239847771997537);
	arr[i++] = _newGrad4(0.5029860367700724, -0.7504883828755602, -0.4004672082940195, 0.15296486218853164);
	arr[i++] = _newGrad4(0.5029860367700724, -0.7504883828755602, 0.15296486218853164, -0.4004672082940195);
	arr[i++] = _newGrad4(0.4553054119602712, -0.8828161875373585, 0.08164729285680945, 0.08164729285680945);
	arr[i++] = _newGrad4(0.8828161875373585, -0.4553054119602712, -0.08164729285680945, -0.08164729285680945);
	arr[i++] = _newGrad4(0.7504883828755602, -0.5029860367700724, -0.15296486218853164, 0.4004672082940195);
	arr[i++] = _newGrad4(0.7504883828755602, -0.5029860367700724, 0.4004672082940195, -0.15296486218853164);
	arr[i++] = _newGrad4(0.6740059517812944, -0.5794684678643381, 0.3239847771997537, 0.3239847771997537);
	arr[i++] = _newGrad4(0.5794684678643381, -0.3239847771997537, -0.6740059517812944, -0.3239847771997537);
	arr[i++] = _newGrad4(0.5029860367700724, -0.4004672082940195, -0.7504883828755602, 0.15296486218853164);
	arr[i++] = _newGrad4(0.5029860367700724, 0.15296486218853164, -0.7504883828755602, -0.4004672082940195);
	arr[i++] = _newGrad4(0.4553054119602712, 0.08164729285680945, -0.8828161875373585, 0.08164729285680945);
	arr[i++] = _newGrad4(0.8828161875373585, -0.08164729285680945, -0.4553054119602712, -0.08164729285680945);
	arr[i++] = _newGrad4(0.7504883828755602, -0.15296486218853164, -0.5029860367700724, 0.4004672082940195);
	arr[i++] = _newGrad4(0.7504883828755602, 0.4004672082940195, -0.5029860367700724, -0.15296486218853164);
	arr[i++] = _newGrad4(0.6740059517812944, 0.3239847771997537, -0.5794684678643381, 0.3239847771997537);
	arr[i++] = _newGrad4(0.5794684678643381, -0.3239847771997537, -0.3239847771997537, -0.6740059517812944);
	arr[i++] = _newGrad4(0.5029860367700724, -0.4004672082940195, 0.15296486218853164, -0.7504883828755602);
	arr[i++] = _newGrad4(0.5029860367700724, 0.15296486218853164, -0.4004672082940195, -0.7504883828755602);
	arr[i++] = _newGrad4(0.4553054119602712, 0.08164729285680945, 0.08164729285680945, -0.8828161875373585);
	arr[i++] = _newGrad4(0.8828161875373585, -0.08164729285680945, -0.08164729285680945, -0.4553054119602712);
	arr[i++] = _newGrad4(0.7504883828755602, -0.15296486218853164, 0.4004672082940195, -0.5029860367700724);
	arr[i++] = _newGrad4(0.7504883828755602, 0.4004672082940195, -0.15296486218853164, -0.5029860367700724);
	arr[i++] = _newGrad4(0.6740059517812944, 0.3239847771997537, 0.3239847771997537, -0.5794684678643381);
	arr[i++] = _newGrad4(0.03381941603233842, 0.03381941603233842, 0.03381941603233842, 0.9982828964265062);
	arr[i++] = _newGrad4(-0.044802370851755174, -0.044802370851755174, 0.508629699630796, 0.8586508742123365);
	arr[i++] = _newGrad4(-0.044802370851755174, 0.508629699630796, -0.044802370851755174, 0.8586508742123365);
	arr[i++] = _newGrad4(-0.12128480194602098, 0.4321472685365301, 0.4321472685365301, 0.7821684431180708);
	arr[i++] = _newGrad4(0.508629699630796, -0.044802370851755174, -0.044802370851755174, 0.8586508742123365);
	arr[i++] = _newGrad4(0.4321472685365301, -0.12128480194602098, 0.4321472685365301, 0.7821684431180708);
	arr[i++] = _newGrad4(0.4321472685365301, 0.4321472685365301, -0.12128480194602098, 0.7821684431180708);
	arr[i++] = _newGrad4(0.37968289875261624, 0.37968289875261624, 0.37968289875261624, 0.753341017856078);
	arr[i++] = _newGrad4(0.03381941603233842, 0.03381941603233842, 0.9982828964265062, 0.03381941603233842);
	arr[i++] = _newGrad4(-0.044802370851755174, 0.044802370851755174, 0.8586508742123365, 0.508629699630796);
	arr[i++] = _newGrad4(-0.044802370851755174, 0.508629699630796, 0.8586508742123365, -0.044802370851755174);
	arr[i++] = _newGrad4(-0.12128480194602098, 0.4321472685365301, 0.7821684431180708, 0.4321472685365301);
	arr[i++] = _newGrad4(0.508629699630796, -0.044802370851755174, 0.8586508742123365, -0.044802370851755174);
	arr[i++] = _newGrad4(0.4321472685365301, -0.12128480194602098, 0.7821684431180708, 0.4321472685365301);
	arr[i++] = _newGrad4(0.4321472685365301, 0.4321472685365301, 0.7821684431180708, -0.12128480194602098);
	arr[i++] = _newGrad4(0.37968289875261624, 0.37968289875261624, 0.753341017856078, 0.37968289875261624);
	arr[i++] = _newGrad4(0.03381941603233842, 0.9982828964265062, 0.03381941603233842, 0.03381941603233842);
	arr[i++] = _newGrad4(-0.044802370851755174, 0.8586508742123365, -0.044802370851755174, 0.508629699630796);
	arr[i++] = _newGrad4(-0.044802370851755174, 0.8586508742123365, 0.508629699630796, -0.044802370851755174);
	arr[i++] = _newGrad4(-0.12128480194602098, 0.7821684431180708, 0.4321472685365301, 0.4321472685365301);
	arr[i++] = _newGrad4(0.508629699630796, 0.8586508742123365, -0.044802370851755174, -0.044802370851755174);
	arr[i++] = _newGrad4(0.4321472685365301, 0.7821684431180708, -0.12128480194602098, 0.4321472685365301);
	arr[i++] = _newGrad4(0.4321472685365301, 0.7821684431180708, 0.4321472685365301, -0.12128480194602098);
	arr[i++] = _newGrad4(0.37968289875261624, 0.753341017856078, 0.37968289875261624, 0.37968289875261624);
	arr[i++] = _newGrad4(0.9982828964265062, 0.03381941603233842, 0.03381941603233842, 0.03381941603233842);
	arr[i++] = _newGrad4(0.8586508742123365, -0.044802370851755174, -0.044802370851755174, 0.508629699630796);
	arr[i++] = _newGrad4(0.8586508742123365, -0.044802370851755174, 0.508629699630796, -0.044802370851755174);
	arr[i++] = _newGrad4(0.7821684431180708, -0.12128480194602098, 0.4321472685365301, 0.4321472685365301);
	arr[i++] = _newGrad4(0.8586508742123365, 0.508629699630796, -0.044802370851755174, -0.044802370851755174);
	arr[i++] = _newGrad4(0.7821684431180708, 0.4321472685365301, -0.12128480194602098, 0.4321472685365301);
	arr[i++] = _newGrad4(0.7821684431180708, 0.4321472685365301, 0.4321472685365301, -0.12128480194602098);
	arr[i++] = _newGrad4(0.753341017856078, 0.37968289875261624, 0.37968289875261624, 0.37968289875261624);
	Grad4 *gradients4D = _newGrad4Arr(PSIZE);
	for (int i = 0; i < 160; i++)
	{
		arr[i].dx /= N4;
		arr[i].dy /= N4;
		arr[i].dz /= N4;
		arr[i].dw /= N4;
	}
	for (int i = 0; i < PSIZE; i++)
	{
		gradients4D[i] = arr[i % 160];
	}
	return gradients4D;
}

LatticePoint2D *_newLatticePoint2D(int xsv, int ysv)
{
	LatticePoint2D *plp2D = (LatticePoint2D *)malloc(sizeof(LatticePoint2D));
	plp2D->xsv = xsv;
	plp2D->ysv = ysv;
	double ssv = (xsv + ysv) * -0.211324865405187;
	plp2D->dx = -xsv - ssv;
	plp2D->dy = -ysv - ssv;
	return plp2D;
}

LatticePoint3D *_newLatticePoint3D(int xrv, int yrv, int zrv, int lattice)
{
	LatticePoint3D *plp3D = (LatticePoint3D *)malloc(sizeof(LatticePoint3D));
	plp3D->dxr = -xrv + lattice * 0.5;
	plp3D->dyr = -yrv + lattice * 0.5;
	plp3D->dzr = -zrv + lattice * 0.5;
	plp3D->xrv = xrv + lattice * 1024;
	plp3D->yrv = yrv + lattice * 1024;
	plp3D->zrv = zrv + lattice * 1024;
	return plp3D;
}

LatticePoint4D *_newLatticePoint4D(int xsv, int ysv, int zsv, int wsv)
{
	LatticePoint4D *plp4D = (LatticePoint4D *)malloc(sizeof(LatticePoint4D));
	plp4D->xsv = xsv + 409;
	plp4D->ysv = ysv + 409;
	plp4D->zsv = zsv + 409;
	plp4D->wsv = wsv + 409;
	double ssv = (xsv + ysv + zsv + wsv) * 0.309016994374947;
	plp4D->dx = -xsv - ssv;
	plp4D->dy = -ysv - ssv;
	plp4D->dz = -zsv - ssv;
	plp4D->dw = -wsv - ssv;
	plp4D->xsi = 0.2 - xsv;
	plp4D->ysi = 0.2 - ysv;
	plp4D->zsi = 0.2 - zsv;
	plp4D->wsi = 0.2 - wsv;
	plp4D->ssiDelta = (0.8 - xsv - ysv - zsv - wsv) * 0.309016994374947;
	return plp4D;
}

LatticePoint2D **_newLatticePoint2DConstArray()
{
	LatticePoint2D **plp2DArr = (LatticePoint2D **)malloc(sizeof(LatticePoint2D *) * 4);
	plp2DArr[0] = _newLatticePoint2D(1, 0);
	plp2DArr[1] = _newLatticePoint2D(0, 0);
	plp2DArr[2] = _newLatticePoint2D(1, 1);
	plp2DArr[3] = _newLatticePoint2D(0, 1);
	return plp2DArr;
}

LatticePoint3D **_newLatticePoint3DConstArray()
{
	LatticePoint3D **plp3DArr = (LatticePoint3D **)malloc(sizeof(LatticePoint3D *) * 8);
	for (int i = 0; i < 8; i++)
	{
		int i1, j1, k1, i2, j2, k2;
		i1 = (i >> 0) & 1;
		j1 = (i >> 1) & 1;
		k1 = (i >> 2) & 1;
		i2 = i1 ^ 1;
		j2 = j1 ^ 1;
		k2 = k1 ^ 1;

		// The two points within this octant, one from each of the two cubic half-lattices.
		LatticePoint3D *c0 = _newLatticePoint3D(i1, j1, k1, 0);
		LatticePoint3D *c1 = _newLatticePoint3D(i1 + i2, j1 + j2, k1 + k2, 1);

		// Each single step away on the first half-lattice.
		LatticePoint3D *c2 = _newLatticePoint3D(i1 ^ 1, j1, k1, 0);
		LatticePoint3D *c3 = _newLatticePoint3D(i1, j1 ^ 1, k1, 0);
		LatticePoint3D *c4 = _newLatticePoint3D(i1, j1, k1 ^ 1, 0);

		// Each single step away on the second half-lattice.
		LatticePoint3D *c5 = _newLatticePoint3D(i1 + (i2 ^ 1), j1 + j2, k1 + k2, 1);
		LatticePoint3D *c6 = _newLatticePoint3D(i1 + i2, j1 + (j2 ^ 1), k1 + k2, 1);
		LatticePoint3D *c7 = _newLatticePoint3D(i1 + i2, j1 + j2, k1 + (k2 ^ 1), 1);

		// First two are guaranteed.
		c0->nextOnFailure = c0->nextOnSuccess = c1;
		c1->nextOnFailure = c1->nextOnSuccess = c2;

		// Once we find one on the first half-lattice, the rest are out.
		// In addition, knowing c2 rules out c5.
		c2->nextOnFailure = c3;
		c2->nextOnSuccess = c6;
		c3->nextOnFailure = c4;
		c3->nextOnSuccess = c5;
		c4->nextOnFailure = c4->nextOnSuccess = c5;

		// Once we find one on the second half-lattice, the rest are out.
		c5->nextOnFailure = c6;
		c5->nextOnSuccess = NULL;
		c6->nextOnFailure = c7;
		c6->nextOnSuccess = NULL;
		c7->nextOnFailure = c7->nextOnSuccess = NULL;

		plp3DArr[i] = c0;
	}
	return plp3DArr;
}

LatticePoint4D **_newLatticePoint4DConstArray()
{
	LatticePoint4D **plp4DArr = (LatticePoint4D **)malloc(sizeof(LatticePoint4D *) * 16);
	for (int i = 0; i < 16; i++)
	{
		plp4DArr[i] = _newLatticePoint4D((i >> 0) & 1, (i >> 1) & 1, (i >> 2) & 1, (i >> 3) & 1);
	}
	return plp4DArr;
}

/*
 * Noise Evaluators
 */

/**
 * 2D Simplex noise base.
 * Lookup table implementation inspired by DigitalShadow.
 */
double _noise2_Base(OpenSimplexEnv *ose, OpenSimplexGradients *osg, double xs, double ys)
{
	double value = 0;

	// Get base points and offsets
	int xsb = _fastFloor(xs), ysb = _fastFloor(ys);
	double xsi = xs - xsb, ysi = ys - ysb;

	// Index to point list
	int index = (int)((ysi - xsi) / 2 + 1);

	double ssi = (xsi + ysi) * -0.211324865405187;
	double xi = xsi + ssi, yi = ysi + ssi;

	// Point contributions
	for (int i = 0; i < 3; i++)
	{
		LatticePoint2D *c = ose->LOOKUP_2D[index + i];

		double dx = xi + c->dx, dy = yi + c->dy;
		double attn = 0.5 - dx * dx - dy * dy;
		if (attn <= 0)
			continue;

		int pxm = (xsb + c->xsv) & PMASK, pym = (ysb + c->ysv) & PMASK;
		Grad2 grad = osg->permGrad2[osg->perm[pxm] ^ pym];
		double extrapolation = grad.dx * dx + grad.dy * dy;

		attn *= attn;
		value += attn * attn * extrapolation;
	}

	return value;
}

/**
 * 2D Simplex noise, standard lattice orientation.
 */
double noise2(OpenSimplexEnv *ose, OpenSimplexGradients *osg, double x, double y)
{

	// Get points for A2* lattice
	double s = 0.366025403784439 * (x + y);
	double xs = x + s, ys = y + s;

	return _noise2_Base(ose, osg, xs, ys);
}

/**
 * 2D Simplex noise, with Y pointing down the main diagonal.
 * Might be better for a 2D sandbox style game, where Y is vertical.
 * Probably slightly less optimal for heightmaps or continent maps.
 */
double noise2_XBeforeY(OpenSimplexEnv *ose, OpenSimplexGradients *osg, double x, double y)
{

	// Skew transform and rotation baked into one.
	double xx = x * 0.7071067811865476;
	double yy = y * 1.224744871380249;

	return _noise2_Base(ose, osg, yy + xx, yy - xx);
}

/**
 * Generate overlapping cubic lattices for 3D Re-oriented BCC noise.
 * Lookup table implementation inspired by DigitalShadow.
 * It was actually faster to narrow down the points in the loop itself,
 * than to build up the index with enough info to isolate 4 points.
 */
double _noise3_BCC(OpenSimplexEnv *ose, OpenSimplexGradients *osg, double xr, double yr, double zr)
{

	// Get base and offsets inside cube of first lattice.
	int xrb = _fastFloor(xr), yrb = _fastFloor(yr), zrb = _fastFloor(zr);
	double xri = xr - xrb, yri = yr - yrb, zri = zr - zrb;

	// Identify which octant of the cube we're in. This determines which cell
	// in the other cubic lattice we're in, and also narrows down one point on each.
	int xht = (int)(xri + 0.5), yht = (int)(yri + 0.5), zht = (int)(zri + 0.5);
	int index = (xht << 0) | (yht << 1) | (zht << 2);

	// Point contributions
	double value = 0;
	LatticePoint3D *c = ose->LOOKUP_3D[index];
	while (c != NULL)
	{
		double dxr = xri + c->dxr, dyr = yri + c->dyr, dzr = zri + c->dzr;
		double attn = 0.5 - dxr * dxr - dyr * dyr - dzr * dzr;
		if (attn < 0)
		{
			c = c->nextOnFailure;
		}
		else
		{
			int pxm = (xrb + c->xrv) & PMASK, pym = (yrb + c->yrv) & PMASK, pzm = (zrb + c->zrv) & PMASK;
			Grad3 grad = osg->permGrad3[osg->perm[osg->perm[pxm] ^ pym] ^ pzm];
			double extrapolation = grad.dx * dxr + grad.dy * dyr + grad.dz * dzr;

			attn *= attn;
			value += attn * attn * extrapolation;
			c = c->nextOnSuccess;
		}
	}
	return value;
}

/**
 * 3D Re-oriented 4-point BCC noise, classic orientation.
 * Proper substitute for 3D Simplex in light of Forbidden Formulae.
 * Use noise3_XYBeforeZ or noise3_XZBeforeY instead, wherever appropriate.
 */
double noise3_Classic(OpenSimplexEnv *ose, OpenSimplexGradients *osg, double x, double y, double z)
{

	// Re-orient the cubic lattices via rotation, to produce the expected look on cardinal planar slices.
	// If texturing objects that don't tend to have cardinal plane faces, you could even remove this.
	// Orthonormal rotation. Not a skew transform.
	double r = (2.0 / 3.0) * (x + y + z);
	double xr = r - x, yr = r - y, zr = r - z;

	// Evaluate both lattices to form a BCC lattice.
	return _noise3_BCC(ose, osg, xr, yr, zr);
}

/**
 * 3D Re-oriented 4-point BCC noise, with better visual isotropy in (X, Y).
 * Recommended for 3D terrain and time-varied animations.
 * The Z coordinate should always be the "different" coordinate in your use case.
 * If Y is vertical in world coordinates, call noise3_XYBeforeZ(x, z, Y) or use noise3_XZBeforeY.
 * If Z is vertical in world coordinates, call noise3_XYBeforeZ(x, y, Z).
 * For a time varied animation, call noise3_XYBeforeZ(x, y, T).
 */
double noise3_XYBeforeZ(OpenSimplexEnv *ose, OpenSimplexGradients *osg, double x, double y, double z)
{

	// Re-orient the cubic lattices without skewing, to make X and Y triangular like 2D.
	// Orthonormal rotation. Not a skew transform.
	double xy = x + y;
	double s2 = xy * -0.211324865405187;
	double zz = z * 0.577350269189626;
	double xr = x + s2 - zz, yr = y + s2 - zz;
	double zr = xy * 0.577350269189626 + zz;

	// Evaluate both lattices to form a BCC lattice.
	return _noise3_BCC(ose, osg, xr, yr, zr);
}

/**
 * 3D Re-oriented 4-point BCC noise, with better visual isotropy in (X, Z).
 * Recommended for 3D terrain and time-varied animations.
 * The Y coordinate should always be the "different" coordinate in your use case.
 * If Y is vertical in world coordinates, call noise3_XZBeforeY(x, Y, z).
 * If Z is vertical in world coordinates, call noise3_XZBeforeY(x, Z, y) or use noise3_XYBeforeZ.
 * For a time varied animation, call noise3_XZBeforeY(x, T, y) or use noise3_XYBeforeZ.
 */
double noise3_XZBeforeY(OpenSimplexEnv *ose, OpenSimplexGradients *osg, double x, double y, double z)
{

	// Re-orient the cubic lattices without skewing, to make X and Z triangular like 2D.
	// Orthonormal rotation. Not a skew transform.
	double xz = x + z;
	double s2 = xz * -0.211324865405187;
	double yy = y * 0.577350269189626;
	double xr = x + s2 - yy;
	double zr = z + s2 - yy;
	double yr = xz * 0.577350269189626 + yy;

	// Evaluate both lattices to form a BCC lattice.
	return _noise3_BCC(ose, osg, xr, yr, zr);
}

/**
 * 4D OpenSimplex2F noise base.
 * Current implementation not fully optimized by lookup tables.
 * But still comes out slightly ahead of Gustavson's Simplex in tests.
 */
double _noise4_Base(OpenSimplexEnv *ose, OpenSimplexGradients *osg, double xs, double ys, double zs, double ws)
{
	double value = 0;

	// Get base points and offsets
	int xsb = _fastFloor(xs), ysb = _fastFloor(ys), zsb = _fastFloor(zs), wsb = _fastFloor(ws);
	double xsi = xs - xsb, ysi = ys - ysb, zsi = zs - zsb, wsi = ws - wsb;

	// If we're in the lower half, flip so we can repeat the code for the upper half. We'll flip back later.
	double siSum = xsi + ysi + zsi + wsi;
	double ssi = siSum * 0.309016994374947; // Prep for vertex contributions.
	bool inLowerHalf = (siSum < 2);
	if (inLowerHalf)
	{
		xsi = 1 - xsi;
		ysi = 1 - ysi;
		zsi = 1 - zsi;
		wsi = 1 - wsi;
		siSum = 4 - siSum;
	}

	// Consider opposing vertex pairs of the octahedron formed by the central cross-section of the stretched tesseract
	double aabb = xsi + ysi - zsi - wsi, abab = xsi - ysi + zsi - wsi, abba = xsi - ysi - zsi + wsi;
	double aabbScore = abs(aabb), ababScore = abs(abab), abbaScore = abs(abba);

	// Find the closest point on the stretched tesseract as if it were the upper half
	int vertexIndex, via, vib;
	double asi, bsi;
	if (aabbScore > ababScore && aabbScore > abbaScore)
	{
		if (aabb > 0)
		{
			asi = zsi;
			bsi = wsi;
			vertexIndex = 0b0011;
			via = 0b0111;
			vib = 0b1011;
		}
		else
		{
			asi = xsi;
			bsi = ysi;
			vertexIndex = 0b1100;
			via = 0b1101;
			vib = 0b1110;
		}
	}
	else if (ababScore > abbaScore)
	{
		if (abab > 0)
		{
			asi = ysi;
			bsi = wsi;
			vertexIndex = 0b0101;
			via = 0b0111;
			vib = 0b1101;
		}
		else
		{
			asi = xsi;
			bsi = zsi;
			vertexIndex = 0b1010;
			via = 0b1011;
			vib = 0b1110;
		}
	}
	else
	{
		if (abba > 0)
		{
			asi = ysi;
			bsi = zsi;
			vertexIndex = 0b1001;
			via = 0b1011;
			vib = 0b1101;
		}
		else
		{
			asi = xsi;
			bsi = wsi;
			vertexIndex = 0b0110;
			via = 0b0111;
			vib = 0b1110;
		}
	}
	if (bsi > asi)
	{
		via = vib;
		double temp = bsi;
		bsi = asi;
		asi = temp;
	}
	if (siSum + asi > 3)
	{
		vertexIndex = via;
		if (siSum + bsi > 4)
		{
			vertexIndex = 0b1111;
		}
	}

	// Now flip back if we're actually in the lower half.
	if (inLowerHalf)
	{
		xsi = 1 - xsi;
		ysi = 1 - ysi;
		zsi = 1 - zsi;
		wsi = 1 - wsi;
		vertexIndex ^= 0b1111;
	}

	// Five points to add, total, from five copies of the A4 lattice.
	for (int i = 0; i < 5; i++)
	{

		// Update xsb/etc. and add the lattice point's contribution.
		LatticePoint4D *c = ose->VERTICES_4D[vertexIndex];
		xsb += c->xsv;
		ysb += c->ysv;
		zsb += c->zsv;
		wsb += c->wsv;
		double xi = xsi + ssi, yi = ysi + ssi, zi = zsi + ssi, wi = wsi + ssi;
		double dx = xi + c->dx, dy = yi + c->dy, dz = zi + c->dz, dw = wi + c->dw;
		double attn = 0.5 - dx * dx - dy * dy - dz * dz - dw * dw;
		if (attn > 0)
		{
			int pxm = xsb & PMASK, pym = ysb & PMASK, pzm = zsb & PMASK, pwm = wsb & PMASK;
			Grad4 grad = osg->permGrad4[osg->perm[osg->perm[osg->perm[pxm] ^ pym] ^ pzm] ^ pwm];
			double ramped = grad.dx * dx + grad.dy * dy + grad.dz * dz + grad.dw * dw;

			attn *= attn;
			value += attn * attn * ramped;
		}

		// Maybe this helps the compiler/JVM/LLVM/etc. know we can end the loop here. Maybe not.
		if (i == 4)
			break;

		// Update the relative skewed coordinates to reference the vertex we just added.
		// Rather, reference its counterpart on the lattice copy that is shifted down by
		// the vector <-0.2, -0.2, -0.2, -0.2>
		xsi += c->xsi;
		ysi += c->ysi;
		zsi += c->zsi;
		wsi += c->wsi;
		ssi += c->ssiDelta;

		// Next point is the closest vertex on the 4-simplex whose base vertex is the aforementioned vertex.
		double score0 = 1.0 + ssi * (-1.0 / 0.309016994374947); // Seems slightly faster than 1.0-xsi-ysi-zsi-wsi
		vertexIndex = 0b0000;
		if (xsi >= ysi && xsi >= zsi && xsi >= wsi && xsi >= score0)
		{
			vertexIndex = 0b0001;
		}
		else if (ysi > xsi && ysi >= zsi && ysi >= wsi && ysi >= score0)
		{
			vertexIndex = 0b0010;
		}
		else if (zsi > xsi && zsi > ysi && zsi >= wsi && zsi >= score0)
		{
			vertexIndex = 0b0100;
		}
		else if (wsi > xsi && wsi > ysi && wsi > zsi && wsi >= score0)
		{
			vertexIndex = 0b1000;
		}
	}

	return value;
}

/**
 * 4D OpenSimplex2F noise, classic lattice orientation.
 */
double noise4_Classic(OpenSimplexEnv *ose, OpenSimplexGradients *osg, double x, double y, double z, double w)
{

	// Get points for A4 lattice
	double s = -0.138196601125011 * (x + y + z + w);
	double xs = x + s, ys = y + s, zs = z + s, ws = w + s;

	return _noise4_Base(ose, osg, xs, ys, zs, ws);
}

/**
 * 4D OpenSimplex2F noise, with XY and ZW forming orthogonal triangular-based planes.
 * Recommended for 3D terrain, where X and Y (or Z and W) are horizontal.
 * Recommended for noise(x, y, sin(time), cos(time)) trick.
 */
double noise4_XYBeforeZW(OpenSimplexEnv *ose, OpenSimplexGradients *osg, double x, double y, double z, double w)
{

	double s2 = (x + y) * -0.178275657951399372 + (z + w) * 0.215623393288842828;
	double t2 = (z + w) * -0.403949762580207112 + (x + y) * -0.375199083010075342;
	double xs = x + s2, ys = y + s2, zs = z + t2, ws = w + t2;

	return _noise4_Base(ose, osg, xs, ys, zs, ws);
}

/**
 * 4D OpenSimplex2F noise, with XZ and YW forming orthogonal triangular-based planes.
 * Recommended for 3D terrain, where X and Z (or Y and W) are horizontal.
 */
double noise4_XZBeforeYW(OpenSimplexEnv *ose, OpenSimplexGradients *osg, double x, double y, double z, double w)
{

	double s2 = (x + z) * -0.178275657951399372 + (y + w) * 0.215623393288842828;
	double t2 = (y + w) * -0.403949762580207112 + (x + z) * -0.375199083010075342;
	double xs = x + s2, ys = y + t2, zs = z + s2, ws = w + t2;

	return _noise4_Base(ose, osg, xs, ys, zs, ws);
}

/**
 * 4D OpenSimplex2F noise, with XYZ oriented like noise3_Classic,
 * and W for an extra degree of freedom. W repeats eventually.
 * Recommended for time-varied animations which texture a 3D object (W=time)
 */
double noise4_XYZBeforeW(OpenSimplexEnv *ose, OpenSimplexGradients *osg, double x, double y, double z, double w)
{

	double xyz = x + y + z;
	double ww = w * 0.2236067977499788;
	double s2 = xyz * -0.16666666666666666 + ww;
	double xs = x + s2, ys = y + s2, zs = z + s2, ws = -0.5 * xyz + ww;

	return _noise4_Base(ose, osg, xs, ys, zs, ws);
}

OpenSimplexEnv *initOpenSimplex()
{
	OpenSimplexEnv *ose = (OpenSimplexEnv *)malloc(sizeof(OpenSimplexEnv));
	ose->GRADIENTS_2D = _newGrad2ConstArray();
	ose->GRADIENTS_3D = _newGrad3ConstArray();
	ose->GRADIENTS_4D = _newGrad4ConstArray();
	ose->LOOKUP_2D = _newLatticePoint2DConstArray();
	ose->LOOKUP_3D = _newLatticePoint3DConstArray();
	ose->VERTICES_4D = _newLatticePoint4DConstArray();
	return ose;
}

OpenSimplexGradients *newOpenSimplexGradients(OpenSimplexEnv *ose, long seed)
{
	OpenSimplexGradients *osg = (OpenSimplexGradients *)malloc(sizeof(OpenSimplexGradients));
	osg->perm = _newShortArr(PSIZE);
	osg->permGrad2 = _newGrad2Arr(PSIZE);
	osg->permGrad3 = _newGrad3Arr(PSIZE);
	osg->permGrad4 = _newGrad4Arr(PSIZE);
	short *source = _newShortArr(PSIZE);
	for (short i = 0; i < PSIZE; i++)
	{
		source[i] = i;
	}
	for (int i = PSIZE - 1; i >= 0; i--)
	{
		seed = seed * 6364136223846793005L + 1442695040888963407L;
		int r = (int)((seed + 31) % (i + 1));
		if (r < 0)
		{
			r += (i + 1);
		}
		osg->perm[i] = source[r];
		osg->permGrad2[i] = ose->GRADIENTS_2D[osg->perm[i]];
		osg->permGrad3[i] = ose->GRADIENTS_3D[osg->perm[i]];
		osg->permGrad4[i] = ose->GRADIENTS_4D[osg->perm[i]];
		source[r] = source[i];
	}
	return osg;
}
