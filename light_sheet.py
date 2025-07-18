#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 14:32:34 2024

@author: brandon
"""

import numpy as np
from scipy.special import jn, j0
from scipy.integrate import quad
from scipy import real, imag
from scipy.optimize import fsolve


def light_sheet_field_intensity(shape, pixel_size, na_obj, na_sheet, lambda_em, lambda_ex, n, dp):
    first_term = compute_first_term(shape, pixel_size, na_obj, lambda_em, n, dp)
    second_term = compute_second_term(shape, pixel_size, na_sheet, lambda_ex, n, dp)
    
    return first_term * second_term


def compute_first_term(shape, pixel_size, na_obj, lambda_em, n, a, dp):
    ygrid, xgrid = np.indices(shape)
    ygrid = (ygrid - shape[0] / 2) * pixel_size
    xgrid = (xgrid - shape[1] / 2) * pixel_size
    
    first_term = np.zeros(shape)
    p = np.arange(0, 1, dp)
    for j in range(shape[0]):
        for i in range(shape[1]):
            y = ygrid[j,i]
            x = xgrid[j,i]

            bessel_arg = 2 * np.pi * na_obj * y * p / lambda_em / n
            bessel_term = j0(bessel_arg)
            trig_arg = np.pi * p ** 2 * x * na_obj ** 2 / (lambda_em * n ** 2)
            trig_term = np.exp(1j * trig_arg)
            integral = np.sum(bessel_term * trig_term * p * dp)
            first_term[j,i] = np.absolute(integral) ** 2

    return first_term
                

def compute_second_term(shape, pixel_size, lambda_ex, n, d, f, dp):
    ygrid, xgrid = np.indices(shape)
    ygrid = (ygrid - shape[0] / 2) * pixel_size
    xgrid = (xgrid - shape[1] / 2) * pixel_size
    
    second_term = np.zeros(shape)
    p = np.arange(0, 1, dp)
    for j in range(shape[0]):
        for i in range(shape[1]):
            y = ygrid[j,i]
            x = xgrid[j,i]
            r = np.sqrt(y ** 2 + x ** 2)

            trig_arg = np.pi * p ** 2 * x * d ** 2 / (f ** 2 * (4 + d ** 2 / f ** 2) * lambda_ex * n ** 2)
            trig_term = (np.cos(trig_arg) + 1j * np.sin(trig_arg))
            integral = np.sum(trig_term * p * dp)
            second_term[j,i] = np.absolute(integral) ** 2
    
    return second_term





"""ported from deconv paper"""
def LsMakePSF(dxy, dz, NA, nf, lambda_ex, lambda_em, fcyl, slitwidth):
    print('calculating PSF...')
    nxy, nz, FWHMxy, FWHMz = DeterminePSFsize(dxy, dz, NA, nf, lambda_ex, lambda_em, fcyl, slitwidth)
    NAls = np.sin(np.arctan(slitwidth / (2 * fcyl)))
    psf = samplePSF(dxy, dz, nxy, nz, NA, nf, lambda_ex, lambda_em, NAls)
    print('ok')
    return psf, nxy, nz, FWHMxy, FWHMz

def DeterminePSFsize(dxy, dz, NA, nf, lambda_ex, lambda_em, fcyl, slitwidth):
    # Size of PSF grid is gridsize (xy z) times FWHM
    gridsizeXY = 2
    gridsizeZ = 2

    NAls = np.sin(np.arctan(0.5 * slitwidth / fcyl))
    halfmax = 0.5 * LsPSFeq(0, 0, 0, NA, nf, lambda_ex, lambda_em, NAls)

    # Find zero crossings
    fxy = lambda x: LsPSFeq(x, 0, 0, NA, nf, lambda_ex, lambda_em, NAls) - halfmax
    fz = lambda x: LsPSFeq(0, 0, x, NA, nf, lambda_ex, lambda_em, NAls) - halfmax

    FWHMxy = 2 * np.abs(fsolve(fxy, 100)[0])
    FWHMz = 2 * np.abs(fsolve(fz, 100)[0])

    Rxy = 0.61 * lambda_em / NA
    dxy_corr = min(dxy, Rxy / 3)

    nxy = int(np.ceil(gridsizeXY * FWHMxy / dxy_corr))
    nz = int(np.ceil(gridsizeZ * FWHMz / dz))

    # Ensure that the grid dimensions are odd
    if nxy % 2 == 0:
        nxy += 1
    if nz % 2 == 0:
        nz += 1

    return nxy, nz, FWHMxy, FWHMz


def samplePSF(dxy, dz, nxy, nz, NA_obj, rf, lambda_ex, lambda_em, NA_ls):
    if nxy % 2 == 0 or nz % 2 == 0:
        raise ValueError('function samplePSF: nxy and nz must be odd!')

    psf = np.zeros(((nxy - 1) // 2 + 1, (nxy - 1) // 2 + 1, (nz - 1) // 2 + 1), dtype='float32')

    for z in range((nz - 1) // 2 + 1):
        for y in range((nxy - 1) // 2 + 1):
            print(y)
            for x in range((nxy - 1) // 2 + 1):
                psf[x, y, z] = LsPSFeq(x * dxy, y * dxy, z * dz, NA_obj, rf, lambda_ex, lambda_em, NA_ls)

    psf = mirror8(psf)

    psf = psf / np.sum(psf)

    return psf


def LsPSFeq(x, y, z, NAobj, n, lambda_ex, lambda_em, NAls):
    return PSF(z, 0, x, NAls, n, lambda_ex) + 0 * PSF(x, y, z, NAobj, n, lambda_em)


def PSF(x, y, z, NA, n, _lambda):
    f2 = lambda p: f1(p, x, y, z, _lambda, NA, n)
    result, _, _ = complex_quad(f2, 0, 1)
    return 4 * np.abs(result)**2


def f1(p, x, y, z, _lambda, NA, n):
    return jn(0, 2 * np.pi * NA * np.sqrt(x**2 + y**2) * p / (_lambda * n)) * \
           np.exp(1j * (-np.pi * p**2 * z * NA**2) / (_lambda * n**2)) * p


def mirror8(p1):
    # Mirrors the content of the first quadrant to all other quadrants
    # to obtain the complete PSF.

    sx = 2 * p1.shape[0] - 1
    sy = 2 * p1.shape[1] - 1
    sz = 2 * p1.shape[2] - 1

    cx = int(np.ceil(sx / 2)) - 1
    cy = int(np.ceil(sy / 2)) - 1
    cz = int(np.ceil(sz / 2)) - 1

    R = np.zeros((sx, sy, sz), dtype='float32')
    print(flip3D(p1, 0, 1, 0).shape)
    R[cx:sx, cy:sy, cz:sz] = p1
    R[cx:sx, 0:cy+1, cz:sz] = flip3D(p1, 0, 1, 0)
    R[0:cx+1, 0:cy+1, cz:sz] = flip3D(p1, 1, 1, 0)
    R[0:cx+1, cy:sy, cz:sz] = flip3D(p1, 1, 0, 0)
    R[cx:sx, cy:sy, 0:cz+1] = flip3D(p1, 0, 0, 1)
    R[cx:sx, 0:cy+1, 0:cz+1] = flip3D(p1, 0, 1, 1)
    R[0:cx+1, 0:cy+1, 0:cz+1] = flip3D(p1, 1, 1, 1)
    R[0:cx+1, cy:sy, 0:cz+1] = flip3D(p1, 1, 0, 1)

    return R


def flip3D(data, x, y, z):
    # Utility function for mirror8
    R = data.copy()
    if x:
        R = np.flip(R, axis=0)
    if y:
        R = np.flip(R, axis=1)
    if z:
        R = np.flip(R, axis=2)
    return R


def complex_quad(func, a, b, **kwargs):
    def real_func(x):
        return real(func(x))
    def imag_func(x):
        return imag(func(x))
    real_integral = quad(real_func, a, b, **kwargs)
    imag_integral = quad(imag_func, a, b, **kwargs)
    return (real_integral[0] + 1j*imag_integral[0], real_integral[1:], imag_integral[1:])