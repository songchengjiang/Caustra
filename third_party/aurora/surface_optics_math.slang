// SPDX-License-Identifier: Apache-2.0
// MaterialX v1.39.5 spectral thin-film and Zeltner LTC sheen reference port.
// Source and changes: ORIGIN.md. No environment/rasterizer dependencies.
#ifndef CAUSTRA_SURFACE_OPTICS_MATH
#define CAUSTRA_SURFACE_OPTICS_MATH
static const int FRESNEL_MODEL_DIELECTRIC=0;
static const int FRESNEL_MODEL_CONDUCTOR=1;
static const int FRESNEL_MODEL_SCHLICK=2;
struct FilmFresnelData {
 int model; float3 ior,extinction,F0,F82,F90;
 float exponent,tf_thickness,tf_ior;
};
float film_square(float x){return x*x;}
float3 film_square(float3 x){return x*x;}
float film_pow6(float x){float x2=x*x;return x2*x2*x2;}
float film_cos(float x){return cos(x);}
float3 film_cos(float3 x){return cos(x);}
float film_atan(float x){return atan(x);}
float3 film_atan(float3 y,float3 x){return atan2(y,x);}
float3 film_f0_to_ior(float3 f0){float3 f=sqrt(clamp(f0,float3(.01),float3(.99)));return (float3(1)+f)/(float3(1)-f);}
// https://renderwonk.com/publications/wp-generalization-adobe/gen-adobe.pdf
float3 film_fresnel_hoffman_schlick(float cosTheta, FilmFresnelData fd)
{
    const float COS_THETA_MAX = 1.0 / 7.0;
    const float COS_THETA_FACTOR = 1.0 / (COS_THETA_MAX * pow(1.0 - COS_THETA_MAX, 6.0));

    float x = clamp(cosTheta, 0.0, 1.0);
    float3 a = lerp(fd.F0, fd.F90, pow(1.0 - COS_THETA_MAX, fd.exponent)) * (float3(1.0) - fd.F82) * COS_THETA_FACTOR;
    return lerp(fd.F0, fd.F90, pow(1.0 - x, fd.exponent)) - a * x * film_pow6(1.0 - x);
}

// https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/
float2 film_fresnel_dielectric_polarized(float cosTheta, float ior)
{
    float cosTheta2 = film_square(clamp(cosTheta, 0.0, 1.0));
    float sinTheta2 = 1.0 - cosTheta2;

    float t0 = max(ior * ior - sinTheta2, 0.0);
    float t1 = t0 + cosTheta2;
    float t2 = 2.0 * sqrt(t0) * cosTheta;
    float Rs = (t1 - t2) / (t1 + t2);

    float t3 = cosTheta2 * t0 + sinTheta2 * sinTheta2;
    float t4 = t2 * sinTheta2;
    float Rp = Rs * (t3 - t4) / (t3 + t4);

    return float2(Rp, Rs);
}

// https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/
void film_fresnel_conductor_polarized(float cosTheta, float3 n, float3 k, out float3 Rp, out float3 Rs)
{
    float cosTheta2 = film_square(clamp(cosTheta, 0.0, 1.0));
    float sinTheta2 = 1.0 - cosTheta2;
    float3 n2 = n * n;
    float3 k2 = k * k;

    float3 t0 = n2 - k2 - float3(sinTheta2);
    float3 a2plusb2 = sqrt(t0 * t0 + 4.0 * n2 * k2);
    float3 t1 = a2plusb2 + float3(cosTheta2);
    float3 a = sqrt(max(0.5 * (a2plusb2 + t0), 0.0));
    float3 t2 = 2.0 * a * cosTheta;
    Rs = (t1 - t2) / (t1 + t2);

    float3 t3 = cosTheta2 * a2plusb2 + float3(sinTheta2 * sinTheta2);
    float3 t4 = t2 * sinTheta2;
    Rp = Rs * (t3 - t4) / (t3 + t4);
}

float3 film_fresnel_conductor(float cosTheta, float3 n, float3 k)
{
    float3 Rp, Rs;
    film_fresnel_conductor_polarized(cosTheta, n, k, Rp, Rs);
    return 0.5 * (Rp  + Rs);
}

// https://belcour.github.io/blog/research/publication/2017/05/01/brdf-thin-film.html
void film_fresnel_conductor_phase_polarized(float cosTheta, float eta1, float3 eta2, float3 kappa2, out float3 phiP, out float3 phiS)
{
    float3 k2 = kappa2 / eta2;
    float3 sinThetaSqr = float3(1.0) - cosTheta * cosTheta;
    float3 A = eta2*eta2*(float3(1.0)-k2*k2) - eta1*eta1*sinThetaSqr;
    float3 B = sqrt(A*A + film_square(2.0*eta2*eta2*k2));
    float3 U = sqrt(max(float3(0.0),(A+B)/2.0));
    float3 V = sqrt(max(float3(0.0),(B-A)/2.0));

    phiS = film_atan(2.0*eta1*V*cosTheta, U*U + V*V - film_square(eta1*cosTheta));
    phiP = film_atan(2.0*eta1*eta2*eta2*cosTheta * (2.0*k2*U - (float3(1.0)-k2*k2) * V),
                   film_square(eta2*eta2*(float3(1.0)+k2*k2)*cosTheta) - eta1*eta1*(U*U+V*V));
}

// https://belcour.github.io/blog/research/publication/2017/05/01/brdf-thin-film.html
float3 film_eval_sensitivity(float opd, float3 shift)
{
    // Use Gaussian fits, given by 3 parameters: val, pos and var
    float phase = 2.0*AURORA_PI * opd;
    float3 val = float3(5.4856e-13, 4.4201e-13, 5.2481e-13);
    float3 pos = float3(1.6810e+06, 1.7953e+06, 2.2084e+06);
    float3 var = float3(4.3278e+09, 9.3046e+09, 6.6121e+09);
    float3 xyz = val * sqrt(2.0*AURORA_PI * var) * film_cos(pos * phase + shift) * exp(- var * phase*phase);
    xyz.x   += 9.7470e-14 * sqrt(2.0*AURORA_PI * 4.5282e+09) * film_cos(2.2399e+06 * phase + shift[0]) * exp(- 4.5282e+09 * phase*phase);
    return xyz / 1.0685e-7;
}

// A Practical Extension to Microfacet Theory for the Modeling of Varying Iridescence
// https://belcour.github.io/blog/research/publication/2017/05/01/brdf-thin-film.html
float3 film_fresnel_airy(float cosTheta, FilmFresnelData fd)
{
    // XYZ to CIE 1931 RGB color space (using neutral E illuminant)
    

    // Assume vacuum on the outside
    float eta1 = 1.0;
    // The caller expresses all IORs relative to the incident medium. Unlike the
    // preview's vacuum-only clamp this also admits a lower-IOR exiting film.
    float eta2 = max(fd.tf_ior, 0.0001);
    float3 eta3 = (fd.model == FRESNEL_MODEL_SCHLICK) ? film_f0_to_ior(fd.F0) : fd.ior;
    float3 kappa3 = (fd.model == FRESNEL_MODEL_SCHLICK) ? float3(0.0) : fd.extinction;
    float cosThetaT = sqrt(max(0.0,1.0 - (1.0 - film_square(cosTheta)) * film_square(eta1 / eta2)));

    // First interface
    float2 R12 = film_fresnel_dielectric_polarized(cosTheta, eta2 / eta1);
    if (cosThetaT <= 0.0)
    {
        // Total internal reflection
        return float3(1.0);
    }
    float2 T121 = float2(1.0) - R12;

    // Second interface
    float3 R23p, R23s;
    if (fd.model == FRESNEL_MODEL_SCHLICK)
    {
        float3 f = film_fresnel_hoffman_schlick(cosThetaT, fd);
        R23p = 0.5 * f;
        R23s = 0.5 * f;
    }
    else
    {
        film_fresnel_conductor_polarized(cosThetaT, eta3 / eta2, kappa3 / eta2, R23p, R23s);
    }

    // Phase shift
    float cosB = film_cos(film_atan(eta2 / eta1));
    float2 phi21 = float2(cosTheta < cosB ? 0.0 : AURORA_PI, AURORA_PI);
    float3 phi23p, phi23s;
    if (fd.model == FRESNEL_MODEL_SCHLICK)
    {
        phi23p = float3((eta3[0] < eta2) ? AURORA_PI : 0.0,
                      (eta3[1] < eta2) ? AURORA_PI : 0.0,
                      (eta3[2] < eta2) ? AURORA_PI : 0.0);
        phi23s = phi23p;
    }
    else
    {
        film_fresnel_conductor_phase_polarized(cosThetaT, eta2, eta3, kappa3, phi23p, phi23s);
    }
    float3 r123p = max(sqrt(R12.x*R23p), 0.0);
    float3 r123s = max(sqrt(R12.y*R23s), 0.0);

    // Iridescence term
    float3 I = float3(0.0);
    float3 Cm, Sm;

    // Optical path difference
    float distMeters = fd.tf_thickness * 1.0e-9;
    float opd = 2.0 * eta2 * cosThetaT * distMeters;

    // Iridescence term using spectral antialiasing for Parallel polarization

    // Reflectance term for m=0 (DC term amplitude)
    float3 Rs = (film_square(T121.x) * R23p) / (float3(1.0) - R12.x*R23p);
    I += R12.x + Rs;

    // Reflectance term for m>0 (pairs of diracs)
    Cm = Rs - T121.x;
    for (int m = 1; m <= 3; m++)
    {
        Cm *= r123p;
        Sm  = 2.0 * film_eval_sensitivity(float(m) * opd, float(m)*(phi23p+float3(phi21.x)));
        I  += Cm*Sm;
    }

    // Iridescence term using spectral antialiasing for Perpendicular polarization

    // Reflectance term for m=0 (DC term amplitude)
    float3 Rp = (film_square(T121.y) * R23s) / (float3(1.0) - R12.y*R23s);
    I += R12.y + Rp;

    // Reflectance term for m>0 (pairs of diracs)
    Cm = Rp - T121.y;
    for (int m = 1; m <= 3; m++)
    {
        Cm *= r123s;
        Sm  = 2.0 * film_eval_sensitivity(float(m) * opd, float(m)*(phi23s+float3(phi21.y)));
        I  += Cm*Sm;
    }

    // Average parallel and perpendicular polarization
    I *= 0.5;

    // Convert back to RGB reflectance
    I = clamp(float3(dot(float3(2.3706743,-0.9000405,-0.4706338),I),dot(float3(-0.5138850,1.4253036,0.0885814),I),dot(float3(0.0052982,-0.0146949,1.0093968),I)), 0.0, 1.0);

    return I;
}


float zeltnerAlbedo(float x,float y) {
 float s=y*(.0206607+1.58491*y)/(.0379424+y*(1.32227+y));
 float m=y*(-.193854+y*(-1.14885+y*(1.7932-.95943*y*y)))/(.046391+y);
 float o=y*(.000654023+(-.0207818+.119681*y)*y)/(1.26264+y*(-1.92021+y));
 float q=(x-m)/s; return saturate(exp(-.5*q*q)/(s*sqrt(2*AURORA_PI))+o);
}
float zeltnerA(float x,float y){return (2.58126*x+.813703*y)*y/(1+.310327*x*x+2.60994*x*y);}
float zeltnerB(float x,float y){return sqrt(max(0.0,1-x))*(y-1)*y*y*y/(.0000254053+1.71228*x-1.71506*x*y+1.34174*y*y);}
void zeltnerBasis(float3 v,float3 n,out float3 x,out float3 y) {
 float3 p=v-n*dot(v,n);float d=dot(p,p);
 x=d>1e-12?p*rsqrt(d):normalize(cross(abs(n.z)<.99?float3(0,0,1):float3(0,1,0),n));y=cross(n,x);
}
float zeltnerPdf(float3 v,float3 l,float3 n,float roughness) {
 float nv=saturate(dot(n,v)),nl=dot(n,l);if(nv<=0||nl<=0)return 0;
 float3 x,y;zeltnerBasis(v,n,x,y);float a=zeltnerA(nv,roughness),b=zeltnerB(nv,roughness);
 float3 w=float3(a*dot(l,x)+b*nl,a*dot(l,y),nl);float d=dot(w,w);
 return max(w.z,0.0)/AURORA_PI*(a*a)/(d*d);
}
float3 zeltnerSample(float3 v,float3 n,float roughness,float2 xi) {
 float r=sqrt(xi.x),phi=2*AURORA_PI*xi.y;
 float3 u=float3(r*cos(phi),r*sin(phi),sqrt(max(0.0,1-xi.x)));
 float nv=saturate(dot(n,v)),a=zeltnerA(nv,roughness),b=zeltnerB(nv,roughness);
 float3 w=normalize(float3((u.x-b*u.z)/a,u.y/a,u.z));
 float3 x,y;zeltnerBasis(v,n,x,y);return x*w.x+y*w.y+n*w.z;
}
#endif
