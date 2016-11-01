#include <functional>

#include "auxiliares.cu"

using namespace std;

// Punteros a memoria global
double *g_datos;
double *g_resp;

double *g_verosimilitud;
double *g_verosimilitudParcial;
double *g_sumaProbabilidades;
double *g_medias;
double *g_pesos;
double *g_covarianzas;

double *g_L;
double *g_logDets;

__global__ void paso_e_cholesky(double *g_covarianzas, double *g_L, const size_t numDimensiones)
{
    const size_t k = blockIdx.z;

    for (size_t j = 0; j < numDimensiones; j++) {
        for (size_t h = 0; h < numDimensiones; h++) {
            g_L[k * numDimensiones * numDimensiones + j * numDimensiones + h] = 0.0;
        }
    }

    for (size_t i = 0; i < numDimensiones; i++) {
        for (size_t j = 0; j < i + 1; j++) {
            double suma = 0.0;

            for (size_t h = 0; h < j; h++) {
                suma += g_L[k * numDimensiones * numDimensiones + i * numDimensiones + h] * g_L[k * numDimensiones * numDimensiones + j * numDimensiones + h];
            }

            g_L[k * numDimensiones * numDimensiones + i * numDimensiones + j] = (i == j) ?
                sqrt(g_covarianzas[k * numDimensiones * numDimensiones + i * numDimensiones + i] - suma) :
                (1.0 / g_L[k * numDimensiones * numDimensiones + j * numDimensiones + j] * (g_covarianzas[k * numDimensiones * numDimensiones + i * numDimensiones + j] - suma));
        }
    }
}

__global__ void paso_e(double *g_L, double *g_logDets, double *g_datos, double *g_pesos, double *g_medias, double *g_resp, const size_t n, size_t const numDimensiones)
{
    const size_t i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const size_t k = blockIdx.z;
    const size_t knd = k * numDimensiones;
    const size_t kndnd = knd * numDimensiones;

    if (i == 0 && threadIdx.x == 0) {
       g_logDets[k] = logaritmoDeterminante(g_L, k, numDimensiones);
    }

    __syncthreads();

    if (i < n) {
        extern __shared__ double sharedData[];
        double *v = (double*) &sharedData[threadIdx.x * numDimensiones];
        double suma = 0.0;
        double tmp;

        for (size_t j = 0; j < numDimensiones; j++) {
            tmp = g_datos[j * n + i] - g_medias[knd + j];

            for (size_t h = 0; h < j; h++) {
                tmp -= g_L[kndnd + j * numDimensiones + h] * v[h];
            }

            v[j] = tmp / g_L[kndnd + j * numDimensiones + j];

            suma += v[j] * v[j];
        }

        g_resp[k * n + i] = -0.5 * (suma + g_logDets[k]) + log(g_pesos[k]);
    }
}

__global__ void paso_e2(double *g_resp, double *g_verosimilitudParcial, const size_t n, const size_t numGaussianas)
{
    const size_t i = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    if (i < n) {
        double suma = 0.0;
        double verosimilitudParcial;
        double maxContribucion = MENOS_INFINITO;

        for (size_t k = 0; k < numGaussianas; k++) {
            if (g_resp[k * n + i] > maxContribucion) {
                maxContribucion = g_resp[k * n + i];
            }
        }

        for (size_t k = 0; k < numGaussianas; k++) {
            suma += exp(g_resp[k * n + i] - maxContribucion);
        }

        verosimilitudParcial = maxContribucion + log(suma);

        for (size_t k = 0; k < numGaussianas; k++) {
            g_resp[k * n + i] = exp(g_resp[k * n + i] - verosimilitudParcial);
        }

        g_verosimilitudParcial[i] = verosimilitudParcial;
    }
}

__global__ void paso_e_verosimilitud(double *g_verosimilitudParcial, double *g_verosimilitud, const size_t n)
{
    extern __shared__ double sharedData[];

    reducir<BLOCK_SIZE>([&] (size_t i) -> double { return g_verosimilitudParcial[i]; },
            [&] () -> double* { return &g_verosimilitud[blockIdx.x]; },
            [&] () -> void { reducirFinal<BLOCK_SIZE>([&] (size_t tid) -> double* { return &g_verosimilitud[tid]; }, [&] () -> double* { return &g_verosimilitud[0]; }, sharedData, gridDim.x); },
            n, sharedData, gridDim.x * gridDim.y * gridDim.z);
}

__global__ void paso_m(double *g_resp, double *g_sumaProbabilidades, double *g_pesos, const size_t n)
{
    extern __shared__ double sharedData[];

    const size_t k = blockIdx.z;

    const size_t numGaussianas = gridDim.z;

    reducir<BLOCK_SIZE>([&] (size_t i) -> double { return g_resp[k * n + i]; },
            [&] () -> double* { return &g_sumaProbabilidades[k * gridDim.x + blockIdx.x]; },
            [&] () -> void {
                for (size_t a = 0; a < numGaussianas; a++) {
                    reducirFinal<BLOCK_SIZE>([&] (size_t tid) -> double* { return &g_sumaProbabilidades[a * gridDim.x + tid]; }, [&] () -> double* { return &g_sumaProbabilidades[a]; }, sharedData, gridDim.x);
                    if (threadIdx.x == 0) g_pesos[a] = g_sumaProbabilidades[a] / n;
                }
            }, n, sharedData, gridDim.x * gridDim.z);
}

__global__ void paso_m2(double *g_resp, double *g_datos, double *g_sumaProbabilidades, double *g_medias, const size_t n)
{
    extern __shared__ double sharedData[];

    const size_t j = blockIdx.y;
    const size_t k = blockIdx.z;

    const size_t numGaussianas = gridDim.z;
    const size_t numDimensiones = gridDim.y;

    reducir<BLOCK_SIZE>([&] (size_t i) -> double { return g_resp[k * n + i] * g_datos[j * n + i]; },
            [&] () -> double* { return &g_medias[k * numDimensiones * gridDim.x + j * gridDim.x + blockIdx.x]; },
            [&] () -> void {
                for (size_t a = 0; a < numGaussianas; a++) {
                    for (size_t b = 0; b < numDimensiones; b++) {
                        reducirFinal<BLOCK_SIZE>([&] (size_t tid) -> double* { return &g_medias[a * numDimensiones * gridDim.x + b * gridDim.x + tid]; }, [&] () -> double* { return &g_medias[a * numDimensiones + b]; }, sharedData, gridDim.x);
                        if (threadIdx.x == 0) g_medias[a * numDimensiones + b] /= g_sumaProbabilidades[a];
                    }
                }
            }, n, sharedData, gridDim.x * gridDim.y * gridDim.z);
}

__global__ void paso_m_covarianzas(double *g_resp, double *g_datos, double *g_medias, double *g_covarianzas, const size_t n, const size_t numDimensiones)
{
    __shared__ double sharedData[BLOCK_SIZE];
    __shared__ size_t numBloques;
    __shared__ size_t j;
    __shared__ size_t h;
    __shared__ size_t k;
    __shared__ size_t kn;
    __shared__ size_t jn;
    __shared__ size_t hn;
    __shared__ size_t knd;
    __shared__ double medias_j;
    __shared__ double medias_h;

    if (threadIdx.x == 0) {
        numBloques = gridDim.x * gridDim.y * gridDim.z;
        j = blockIdx.y / numDimensiones;
        h = blockIdx.y % numDimensiones;
        k = blockIdx.z;
        kn = k * n;
        jn = j * n;
        hn = h * n;
        knd = k * numDimensiones;
        medias_j = g_medias[knd + j];
        medias_h = g_medias[knd + h];
    }

    __syncthreads();

    reducir<BLOCK_SIZE>([&] (size_t i) -> double { return g_resp[kn + i] * (g_datos[jn + i] - medias_j) * (g_datos[hn + i] - medias_h); },
            [&] () -> double* { return &g_covarianzas[knd * numDimensiones * gridDim.x + j * numDimensiones * gridDim.x + h * gridDim.x + blockIdx.x]; },
            [&] () -> void {
            }, n, sharedData, numBloques);
}

__global__ void paso_m_covarianzas_final(double *g_sumaProbabilidades, double *g_covarianzas, const size_t numTrozos)
{
    extern __shared__ double sharedData[];

    const size_t j = blockIdx.x;
    const size_t h = blockIdx.y;
    const size_t k = blockIdx.z;

    const size_t numDimensiones = gridDim.y;

    reducirFinal<BLOCK_SIZE>([&] (size_t tid) -> double* { return &g_covarianzas[k * numDimensiones * numDimensiones * numTrozos + j * numDimensiones * numTrozos + h * numTrozos + tid]; }, [&] () -> double* { return &g_covarianzas[k * numDimensiones * numDimensiones + j * numDimensiones + h]; }, sharedData, numTrozos);
    if (threadIdx.x == 0) g_covarianzas[k * numDimensiones * numDimensiones + j * numDimensiones + h] /= g_sumaProbabilidades[k];
}
