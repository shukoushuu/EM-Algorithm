#include <iomanip>
#include <iostream>

#include "kernels.cu"
#include "gmm.h"

using namespace std;

GMM::GMM(size_t numGaussianas, double* pesos, double* medias, double* covarianzas,
         unsigned int maxIteraciones = 250, double precision = 1e-5)
{
    if (pesos == NULL || medias == NULL || covarianzas == NULL) {
        cerr << "ERROR: puntero NULL en la inicializacion.\n"; // 初始化使用了空指针
        throw 1;
    }

    if (numGaussianas < 1) {
        cerr << "ERROR: debe haber al menos 1 gaussiana.\n"; // 至少要有1个高斯
        throw 1;
    }

    this->numGaussianas = numGaussianas; // 高斯mixture数量

    this->pesos = pesos; // peso=weight 各mixture权重
    this->medias = medias; // 各mixture均值
    this->covarianzas = covarianzas; // 各mixture协方差矩阵

    this->datos = NULL; // 输入的数据
    this->maxIteraciones = maxIteraciones; // 最大迭代次数
    this->precision = precision; // 停止精度

    this->verosimilitud = MENOS_INFINITO; // 似然初始化为负无穷
}

GMM::~GMM()
{
    this->datos = NULL;
    this->pesos = NULL;
    this->medias = NULL;
    this->covarianzas = NULL;

    limpiarGPU(); // limpiar=clean, 释放GPU上分配的内存 
}

resultado_type GMM::estimar(double *datos, size_t numMuestras, size_t numDimensiones)
{
    if (datos == NULL || numMuestras < 1 || numDimensiones < 1) {
        cerr << "ERROR: no se han cargado los datos.\n"; // Data not uploaded未载入数据
        throw 1;
    }

    unsigned int i;
    bool finalizado = false; // finalizado=finalized 迭代是否停止
    double ultimaVerosimilitud = verosimilitud; // 最后一次迭代的似然

    this->datos = datos;
    this->numMuestras = numMuestras;
    this->numDimensiones = numDimensiones;

    numRespuestasGPU = numMuestras / (BLOCK_SIZE << 1); // BLOCK_SIZE=128 // 需要的BLOCK数量, 先除以每个BLOCK处理的元素个数取整

    if (numMuestras % (BLOCK_SIZE << 1)) { // 需要的BLOCK数量, 取整有余数的话再+1
        numRespuestasGPU++;
    }

    inicializarGPU();

    for (i = 1; i <= maxIteraciones; i++) {
        if (i % 10 == 1) { // 每10次循环输出一次
            cerr << ".";
        }

        calcular(); // 迭代中执行具体计算的函数

        // Test de convergencia // 测试是否收敛: 本次迭代中似然的相对变化是否小于停止精度
        if (abs((verosimilitud - ultimaVerosimilitud) / verosimilitud) <= precision) {
            finalizado = true;

            break;
        }

        ultimaVerosimilitud = verosimilitud;
    }

    // Asegurar que guardamos el numero de iteracion real
    if (!finalizado) i--;

    // Imprimir ultima iteracion // imprimir=print输出最后一次迭代
    imprimir(finalizado, i);

    resultado_type resultados;

    resultados.iteracion = i;
    resultados.verosimilitud = verosimilitud;

    return resultados;
}

void GMM::calcular() // calcular=calculate
{
    dim3 dimBlock(BLOCK_SIZE, 1, 1); // BLOCK的维度
    size_t arraySize = BLOCK_SIZE * sizeof(double); // 共享内存大小

    // Paso E
    {
        dim3 dimGrid(1, 1, numGaussianas);
        dim3 dimBlock(1, 1, 1); // 因为cholesky分解不好并行, 所以每个block只包含一个thread
        paso_e_cholesky<<<dimGrid, dimBlock, arraySize>>>(g_covarianzas, g_L, numDimensiones); // cholesky分解, 将对称正定的矩阵分解为一个下三角矩阵L及其转置的乘积. 这里分解的是协方差矩阵
    }

    {
        dim3 dimGrid(numRespuestasGPU * 2, 1, numGaussianas); // numRespuestasGPU * 2=样本数
        size_t arraySize = BLOCK_SIZE * numDimensiones * sizeof(double);
        paso_e<<<dimGrid, dimBlock, arraySize>>>(g_L, g_logDets, g_datos, g_pesos, g_medias, g_resp, numMuestras, numDimensiones);
    } // 求log[N(x_i|mu_k, Sigma_k) * P(k)] (不包括d/2log(2*pi)那一项) 存储于g_resp

    {
        dim3 dimGrid(numRespuestasGPU * 2, 1, 1); // numRespuestasGPU * 2=样本数 因为要对每一个点n, 找到使其p_nk最大的那个mixture, 所以每个thread都要遍历k个mixture
        paso_e2<<<dimGrid, dimBlock, arraySize>>>(g_resp, g_verosimilitudParcial, numMuestras, numGaussianas); // 求p_ik存储于g_resp, 求log(Sigma(exp(z_k)))存储于g_verosimilitudParcial
    }

    {
        dim3 dimGrid(numRespuestasGPU, 1, 1);
        paso_e_verosimilitud<<<dimGrid, dimBlock, arraySize>>>(g_verosimilitudParcial, g_verosimilitud, numMuestras); // n个样本的log(Sigma(exp(z_k)))规约获得最终的log(Likelihood)
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

    cudaMalloc(&g_L, numGaussianas * numDimensiones * numDimensiones * sizeof(double)); // 注意: g_L元素个数是numGaussianas * numDimensiones * numDimensiones, 而CPU版中L元素个数是numDimensiones * numDimensiones
    cudaMalloc(&g_logDets, numGaussianas * sizeof(double));

    cudaMalloc(&g_resp, numGaussianas * numMuestras * sizeof(double));

    cudaMemcpy(g_pesos, pesos, numGaussianas * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(g_medias, medias, numGaussianas * numDimensiones * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(g_covarianzas, covarianzas, numGaussianas * numDimensiones * numDimensiones * sizeof(double), cudaMemcpyHostToDevice);

    cudaThreadSynchronize();
}

void GMM::copiarEnGPU() // copy to GPU
{
    cudaThreadSynchronize();
}

void GMM::copiarDesdeGPU() // copy from GPU 从GPU拷贝到CPU
{
    cudaThreadSynchronize();

    cudaMemcpy(medias, g_medias, numGaussianas * numDimensiones * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(pesos, g_pesos, numGaussianas * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&verosimilitud, g_verosimilitud, sizeof(double), cudaMemcpyDeviceToHost);
}

// limpiar=clean, 释放GPU上分配的内存
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


void GMM::imprimir(bool finalizado, unsigned int iteracion) // imprimir=print输出
{
    cerr << setprecision(5) << scientific; // 指数形式输出

    cerr << "\n\n";

    if (finalizado) {
        cerr << "Converge ";
    } else {
        cerr << "No converge ";
    }

    cerr << "en " << iteracion << " iteraciones. Verosimilitud: " << verosimilitud << endl;
}
