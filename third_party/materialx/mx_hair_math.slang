// SPDX-License-Identifier: Apache-2.0
// Adapted from AcademySoftwareFoundation/MaterialX v1.39.5
// libraries/pbrlib/genglsl/mx_chiang_hair_bsdf.glsl.
// GLSL type spellings and helper prefixes converted to Slang. Formulas retained.
// See LICENSE and ORIGIN.md in this directory for upstream attribution.
#ifndef CAUSTRA_OFFICIAL_HAIR_MATH
#define CAUSTRA_OFFICIAL_HAIR_MATH
static const float OFFICIAL_PI = 3.14159265358979323846f;
// https://eugenedeon.com/pdfs/egsrhair.pdf
void official_deon_hair_absorption_from_melanin(
    float melanin_concentration,
    float melanin_redness,
    // constants converted to color via exp(-c). the defaults are lin_rec709 colors, they may be
    // transformed to scene-linear rendering color space.
    float3 eumelanin_color,    // default: (0.657704, 0.498077, 0.254106) == exp(-(0.419, 0.697, 1.37))
    float3 pheomelanin_color,  // default: (0.829443, 0.670320, 0.349937) == exp(-(0.187, 0.4, 1.05))
    out float3 absorption)
{
    float melanin = -log(max(1.0 - melanin_concentration, 0.0001));
    float eumelanin = melanin * (1.0 - melanin_redness);
    float pheomelanin = melanin * melanin_redness;
    absorption = max(
        eumelanin * -log(eumelanin_color) + pheomelanin * -log(pheomelanin_color), 
        float3(0.0)
    );
}

// https://media.disneyanimation.com/uploads/production/publication_asset/152/asset/eurographics2016Fur_Smaller.pdf
void official_chiang_hair_absorption_from_color(float3 color, float betaN, out float3 absorption)
{
    float b2 = betaN* betaN;
    float b4 = b2 * b2;
    float b_fac = 
        5.969 - 
        (0.215 * betaN) + 
        (2.532 * b2) -
        (10.73 * b2 * betaN) + 
        (5.574 * b4) +
        (0.245 * b4 * betaN);
    float3 sigma = log(min(max(color, 0.001), float3(1.0))) / b_fac;
    absorption = sigma * sigma;
}

void official_chiang_hair_roughness(
    float longitudinal,
    float azimuthal,
    float scale_TT,   // empirical roughness scale from Marschner et al. (2003).
    float scale_TRT,  // default: scale_TT = 0.5, scale_TRT = 2.0
    out float2 roughness_R,
    out float2 roughness_TT,
    out float2 roughness_TRT
)
{
    float lr = clamp(longitudinal, 0.001, 1.0);
    float ar = clamp(azimuthal, 0.001, 1.0);

    // longitudinal variance
    float v = 0.726 * lr + 0.812 * lr * lr + 3.7 * pow(lr, 20.0);
    v = v * v;

    float s = 0.265 * ar + 1.194 * ar * ar + 5.372 * pow(ar, 22.0);

    roughness_R = float2(v, s);
    roughness_TT = float2(v * scale_TT * scale_TT, s);
    roughness_TRT = float2(v * scale_TRT * scale_TRT, s);
}

float official_hair_transform_sin_cos(float x)
{
    return sqrt(max(1.0 - x * x, 0.0));
}

float official_hair_I0(float x)
{
    float v = 1.0;
    float n = 1.0;
    float d = 1.0;
    float f = 1.0;
    float x2 = x * x;
    for (int i = 0; i < 9 ; ++i)
    {
        d *= 4.0 * (f * f);
        n *= x2;
        v += n / d;
        f += 1.0;
    }
    return v;
}

float official_hair_log_I0(float x)
{
    if (x > 12.0)
        return x + 0.5 * (-log(2.0 * OFFICIAL_PI) + log(1.0 / x) + 1.0 / (8.0 * x));
    else
        return log(official_hair_I0(x));
}

float official_hair_logistic(float x, float s)
{
    if (x > 0.0)
        x = -x;
    float f = exp(x / s);
    return f / (s * (1.0 + f) * (1.0 + f));
}

float official_hair_logistic_cdf(float x, float s)
{
    return 1.0 / (1.0 + exp(-x / s));
}

float official_hair_trimmed_logistic(float x, float s, float a, float b)
{
    // the constant can be found in Chiang et al. (2016) Appendix A, eq. (12)
    s *= 0.626657;  // sqrt(OFFICIAL_PI/8)
    return official_hair_logistic(x, s) / (official_hair_logistic_cdf(b, s) - official_hair_logistic_cdf(a, s));
}

float official_hair_phi(int p, float gammaO, float gammaT)
{
    float fP = float(p);
    return 2.0 * fP * gammaT - 2.0 * gammaO + fP * OFFICIAL_PI;
}

float official_hair_longitudinal_scattering(  // Mp
    float sinThetaI,
    float cosThetaI,
    float sinThetaO,
    float cosThetaO,
    float v
)
{
    float inv_v = 1.0 / v;
    float a = cosThetaO * cosThetaI * inv_v;
    float b = sinThetaO * sinThetaI * inv_v;
    if (v < 0.1)
        return exp(official_hair_log_I0(a) - b - inv_v + 0.6931 + log(0.5 * inv_v));
    else
        return ((exp(-b) * official_hair_I0(a)) / (2.0 * v * sinh(inv_v)));
}

float official_hair_azimuthal_scattering(  // Np
    float phi,
    int p,
    float s,
    float gammaO,
    float gammaT
)
{
    if (p >= 3)
        return float(0.5 / OFFICIAL_PI);

    float dphi = phi - official_hair_phi(p, gammaO, gammaT);
    if (isinf(dphi))
        return float(0.5 / OFFICIAL_PI);

    while (dphi > OFFICIAL_PI)    dphi -= (2.0 * OFFICIAL_PI);
    while (dphi < (-OFFICIAL_PI)) dphi += (2.0 * OFFICIAL_PI);

    return official_hair_trimmed_logistic(dphi, s, -OFFICIAL_PI, OFFICIAL_PI);
}

void official_hair_alpha_angles(
    float alpha,
    float sinThetaI,
    float cosThetaI,
    out float2 angles[4]
)
{
    // 0:R, 1:TT, 2:TRT, 3:TRRT+
    for (int i = 0; i <= 3; ++i)
    {
        if (alpha == 0.0 || i == 3)
            angles[i] = float2(sinThetaI, cosThetaI);
        else
        {
            float m = 2.0 - float(i) * 3.0;
            float sa = sin(m * alpha);
            float ca = cos(m * alpha);
            angles[i].x = sinThetaI * ca + cosThetaI * sa;
            angles[i].y = cosThetaI * ca - sinThetaI * sa;
        }
    }
}

void official_hair_attenuation(float f, float3 T, out float3 Ap[4])  // Ap
{
    // 0:R, 1:TT, 2:TRT, 3:TRRT+
    Ap[0] = float3(f);
    Ap[1] = (1.0 - f) * (1.0 - f) * T;
    Ap[2] = Ap[1] * T * f;
    Ap[3] = Ap[2] * T * f / (float3(1.0) - T * f);
}


#endif
