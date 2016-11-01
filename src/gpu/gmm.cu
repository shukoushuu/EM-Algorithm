#include <iomanip>
#include <iostream>

#include "kernels.cu"
#include "gmm.h"

using namespace std;

GMM::GMM(size_t numGaussianas, double* pesos, double* medias, double* covarianzas,
         unsigned int maxIteraciones = 250, double precision = 1e-5)
{
    if (pesos == NULL || medias == NULL || covarianzas == NULL) {
        cerr << "ERROR: puntero NULL en la inicializacion.\n";
        throw 1;
    }

    if (numGaussianas < 1) {
        cerr << "ERROR: debe haber al menos 1 gaussiana.\n";
        throw 1;
    }

    this->numGaussianas = numGaussianas;

    this->pesos = pesos;
    this->medias = medias;
    this->covarianzas = covarianzas;

    this->datos = NULL;
    this->maxIteraciones = maxIteraciones;
    this->precision = precision;

    this->verosimilitud = MENOS_INFINITO;
}

GMM::~GMM()
{
    this->datos = NULL;
    this->pesos = NULL;
    this->medias = NULL;
    this->covarianzas = NULL;

    limpiarGPU();
}

resultado_type GMM::estimar(double *datos, size_t numMuestras, size_t numDimensiones)
{
    if (datos == NULL || numMuestras < 1 || numDimensiones < 1) {
        cerr << "ERROR: no se han cargado los datos.\n";
        throw 1;
    }

    unsigned int i;
    bool finalizado = false;
    double ultimaVerosimilitud = verosimilitud;

    this->datos = datos;
    this->numMuestras = numMuestras;
    this->numDimensiones = numDimensiones;

    numRespuestasGPU = numMuestras / (BLOCK_SIZE << 1);

    if (numMuestras % (BLOCK_SIZE << 1)) {
        numRespuestasGPU++;
    }

    inicializarGPU();

    for (i = 1; i <= maxIteraciones; i++) {
        if (i % 10 == 1) {
            cerr << ".";
        }

        calcular();

        // Test de convergencia
        if (abs((verosimilitud - ultimaVerosimilitud) / verosimilitud) <= precision) {
            finalizado = true;

            break;
        }

        ultimaVerosimilitud = verosimilitud;
    }

    // Asegurar que guardamos el numero de iteracion real
    if (!finalizado) i--;

    // Imprimir ultima iteracion
    imprimir(finalizado, i);

    resultado_type resultados;

    resultados.iteracion = i;
    resultados.verosimilitud = verosimilitud;

    return resultados;
}

void GMM::calcular()
{
    dim3 dimBlock(BLOCK_SIZE, 1, 1);
    size_t arraySize = BLOCK_SIZE * sizeof(double);

    // Paso E
    {
        dim3 dimGrid(1, 1, numGaussianas);
        dim3 dimBlock(1, 1, 1);
        paso_e_cholesky<<<dimGrid, dimBlock, arraySize>>>(g_covarianzas, g_L, numDimensiones);
    }

    {
        dim3 dimGrid(numRespuestasGPU * 2, 1, numGaussianas);
        size_t arraySize = BLOCK_SIZE * numDimensiones * sizeof(double);
        paso_e<<<dimGrid, dimBlock, arraySize>>>(g_L, g_logDets, g_datos, g_pesos, g_medias, g_resp, numMuestras, numDimensiones);
    }

    {
        dim3 dimGrid(numRespuestasGPU * 2, 1, 1);
        paso_e2<<<dimGrid, dimBlock, arraySize>>>(g_resp, g_verosimilitudParcial, numMuestras, numGaussianas);
    }

    {
        dim3 dimGrid(numRespuestasGPU, 1, 1);
        paso_e_verosimilitud<<<dimGrid, dimBlock, arraySize>>>(g_verosimilitudParcial, g_verosimilitud, numMuestras);
    }

    // Paso M
    {
        dim3 dimGrid(numRespuestasGPU, 1, numGaussianas);
        paso_m<<<dimGrid, dimBlock, arraySize>>>(g_resp, g_sumaProbabilidades, g_pesos, numMuestras);
    }

    {
        dim3 dimGrid(numRespuestasGPU, numDimensiones, numGaussianas);
        paso_m2<<<dimGrid, dimBlock, arraySize>>>(g_resp, g_datos, g_sumaProbabilidades, g_medias, numMuestras);
    }

    {
        dim3 dimGrid(numRespuestasGPU, numDimensiones * numDimensiones, numGaussianas);
        paso_m_covarianzas<<<dimGrid, dimBlock, arraySize>>>(g_resp, g_datos, g_medias, g_covarianzas, numMuestras, numDimensiones);
    }

    {
        dim3 dimGrid(numDimensiones, numDimensiones, numGaussianas);
        paso_m_covarianzas_final<<<dimGrid, dimBlock, arraySize>>>(g_sumaProbabilidades, g_covarianzas, numRespuestasGPU);
    }

    copiarDesdeGPU();
}

void GMM::inicializarGPU()
{
    cudaMalloc(&g_datos, numMuestras * numDimensiones * sizeof(double));
    cudaMemcpy(g_datos, datos, numMuestras * numDimensiones * sizeof(double), cudaMemcpyHostToDevice);

    cudaMalloc(&g_verosimilitudParcial, numMuestras * sizeof(double));

    cudaMalloc(&g_verosimilitud, numRespuestasGPU * sizeof(double));
    cudaMalloc(&g_sumaProbabilidades, numGaussianas * numRespuestasGPU * sizeof(double));
    cudaMalloc(&g_medias, numGaussianas * numDimensiones * numRespuestasGPU * sizeof(double));
    cudaMalloc(&g_pesos, numGaussianas * sizeof(double));
    cudaMalloc(&g_covarianzas, numGaussianas * numDimensiones * numDimensiones * numRespuestasGPU * sizeof(double));

    cudaMalloc(&g_L, numGaussianas * numDimensiones * numDimensiones * sizeof(double));
    cudaMalloc(&g_logDets, numGaussianas * sizeof(double));

    cudaMalloc(&g_resp, numGaussianas * numMuestras * sizeof(double));

    cudaMemcpy(g_pesos, pesos, numGaussianas * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(g_medias, medias, numGaussianas * numDimensiones * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(g_covarianzas, covarianzas, numGaussianas * numDimensiones * numDimensiones * sizeof(double), cudaMemcpyHostToDevice);

    cudaThreadSynchronize();
}

void GMM::copiarEnGPU()
{
    cudaThreadSynchronize();
}

void GMM::copiarDesdeGPU()
{
    cudaThreadSynchronize();

    cudaMemcpy(medias, g_medias, numGaussianas * numDimensiones * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(pesos, g_pesos, numGaussianas * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&verosimilitud, g_verosimilitud, sizeof(double), cudaMemcpyDeviceToHost);
}

void GMM::limpiarGPU()
{
    cudaFree(g_resp);
    cudaFree(g_datos);

    cudaFree(g_verosimilitud);
    cudaFree(g_verosimilitudParcial);
    cudaFree(g_sumaProbabilidades);
    cudaFree(g_medias);
    cudaFree(g_pesos);
    cudaFree(g_covarianzas);

    cudaFree(g_L);
    cudaFree(g_logDets);
}

void GMM::imprimir(bool finalizado, unsigned int iteracion)
{
    cerr << setprecision(5) << scientific;

    cerr << "\n\n";

    if (finalizado) {
        cerr << "Converge ";
    } else {
        cerr << "No converge ";
    }

    cerr << "en " << iteracion << " iteraciones. Verosimilitud: " << verosimilitud << endl;
}
