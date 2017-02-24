#include <cmath>
#include <ctime>
#include <limits>
#include <iomanip>
#include <iostream>

#include "gsl/gsl_math.h"
#include "gsl/gsl_sf_log.h"

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

    delete [] this->resp;
    delete [] this->L;
    delete [] this->v;
}

resultado_type GMM::estimar(double *datos, size_t numMuestras, size_t numDimensiones) // estimar = estimate
{
    if (datos == NULL || numMuestras < 1 || numDimensiones < 1) {
        cerr << "ERROR: no se han cargado los datos.\n"; // Data not uploaded未载入数据
        throw 1;
    }

    unsigned int i;
    bool finalizado = false; // finalizado=finalized 迭代是否停止
    double ultimaVerosimilitud = verosimilitud; // 最后一次迭代的似然

    this->resp = new double[numMuestras * numGaussianas];
    this->L = new double[numDimensiones * numDimensiones];
    this->v = new double[numDimensiones];

    this->datos = datos;
    this->numMuestras = numMuestras;
    this->numDimensiones = numDimensiones;

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
    double *logDets = new double[numGaussianas];
    double *u = new double[numDimensiones]; // 暂存(x_i - mu_k)
    double suma, verosimilitudParcial, maxContribucion, sumaProbabilidades;

    // Paso E // E step 求log[N(x_n|mu_k, Sigma_k) * P(k)] (不包括d/2log(2*pi)那一项)
    for (size_t k = 0; k < numGaussianas; k++) {
        cholesky(k); // cholesky分解, 将对称正定的矩阵分解为一个下三角矩阵L及其转置的乘积. 这里分解的是协方差矩阵
        logDets[k] = logaritmoDeterminante(); // logarithmDeterminant 协方差矩阵Sigma行列式对数

        for (size_t i = 0; i < numMuestras; i++) { // 计算公式3.7: log [N(x|mu, sigma) * P(k)]
            for (size_t j = 0; j < numDimensiones; j++) {
                u[j] = datos[i * numDimensiones + j] - medias[k * numDimensiones + j]; // 计算(x_i - mu_k)
            }

            resolver(u); // 求v=L^(-1)(x_i - mu_k)

            suma = 0.0;

            for (size_t j = 0; j < numDimensiones; j++) {
                suma += v[j] * v[j];
            }

            resp[i * numGaussianas + k] = -0.5 * (suma + logDets[k]) + gsl_sf_log(pesos[k]); // log[N(x_n|mu_k, Sigma_k) * P(k)] (不包括d/2log(2*pi)那一项)
        }
    } // E step结束

    verosimilitud = 0.0; // verosimilitud=likelihood, 似然

    for (size_t i = 0; i < numMuestras; i++) { // numMuestras待分类的样本数量, 对每一个点n, 找到使其p_nk最大的那个mixture
        maxContribucion = MENOS_INFINITO; // 对于给定的n, 找到使p_nk最大的k, 先初始化为无穷小

        for (size_t k = 0; k < numGaussianas; k++) {
            if (resp[i * numGaussianas + k] > maxContribucion) {
                maxContribucion = resp[i * numGaussianas + k];
            }
        }

        suma = 0.0;

        for (size_t k = 0; k < numGaussianas; k++) {
            suma += exp(resp[i * numGaussianas + k] - maxContribucion); // resp[i * numGaussianas + k]也就是log[N(x_n|mu_k, Sigma_k) * P(k)] (不包括d/2log(2*pi)那一项), 对应下面的z_k
        }

        verosimilitudParcial = maxContribucion + gsl_sf_log(suma); // 根据log-sum-exp公式 log(Sigma(exp(z_k))) = z_max + log(Sigma(exp(z_k - z_max))) verosimilitudParcial = log{Sigma[N(x_n|mu_k, Sigma_k) * P(k)]} (不包括d/2log(2*pi)那一项)

        for (size_t k = 0; k < numGaussianas; k++) {
            resp[i * numGaussianas + k] = exp(resp[i * numGaussianas + k] - verosimilitudParcial); // resp[i * numGaussianas + k]重新赋值为[N(x_n|mu_k, Sigma_k) * P(k)] /Sigma[N(x_n|mu_k, Sigma_k) * P(k)], 这时才是真正的p_nk
        }

        verosimilitud += verosimilitudParcial; // 最终求得log(Likelihood)
    }

    // Paso M // M step
    for (size_t k = 0; k < numGaussianas; k++) { // 对于每个mixture
        sumaProbabilidades = 0.0;

        for (size_t i = 0; i < numMuestras; i++) {
            sumaProbabilidades += resp[i * numGaussianas + k]; // 给定k, 计算所有p_nk之和, 也就是所有采样属于第k个mixture的概率之和
        }

        pesos[k] = sumaProbabilidades / numMuestras; // 公式3.6 对各mixture的权重进行更新

        for (size_t j = 0; j < numDimensiones; j++) {
            suma = 0.0;

            for (size_t i = 0; i < numMuestras; i++) {
                suma += resp[i * numGaussianas + k] * datos[i * numDimensiones + j]; // Sigma(p_nk * x_n)
            }

            medias[k * numDimensiones + j] = suma / sumaProbabilidades; // 更新均值

            for (size_t h = 0; h < numDimensiones; h++) {
                suma = 0.0;

                for (size_t i = 0; i < numMuestras; i++) {
                    suma += resp[i * numGaussianas + k] * (datos[i * numDimensiones + j] - medias[k * numDimensiones + j])
                        * (datos[i * numDimensiones + h] - medias[k * numDimensiones + h]);
                }

                covarianzas[k * numDimensiones * numDimensiones + j * numDimensiones + h] = suma / sumaProbabilidades; // 更新协方差矩阵
            }
        }
    } // M step结束

    // Limpiamos la memoria
    delete [] logDets;
    delete [] u;
}

void GMM::cholesky(size_t gaussiana) // cholesky分解, 将对称正定的矩阵分解为一个下三角矩阵L及其转置的乘积. 这里分解的是协方差矩阵
{
    for (size_t i = 0; i < numDimensiones; i++) { // 先初始化为0
        for (size_t j = 0; j < numDimensiones; j++) {
            L[i * numDimensiones + j] = 0.0;
        }
    }

    for (size_t i = 0; i < numDimensiones; i++) {
        for (size_t j = 0; j < i + 1; j++) { // 只处理下三角的元素(包括对角线上的元素)
            double suma = 0;

            for (size_t k = 0; k < j; k++) {
                suma += L[i * numDimensiones + k] * L[j * numDimensiones + k];
            }

            L[i * numDimensiones + j] = (i == j) ?
                sqrt(covarianzas[gaussiana * numDimensiones * numDimensiones + i * numDimensiones + i] - suma) :
                (1.0 / L[j * numDimensiones + j] * (covarianzas[gaussiana * numDimensiones * numDimensiones + i * numDimensiones + j] - suma));
        }
    }
}

void GMM::resolver(double *muestra) // 求v=L^(-1)(x_i - mu_k)
{
    double suma;

    for (size_t i = 0; i < numDimensiones; i++) { // 先将v初始化为0
        v[i] = 0.0;
    }

    for (size_t i = 0; i < numDimensiones; i++) { //L*y = x - mu， 求解y
        suma = muestra[i];

        for (size_t j = 0; j < i; j++) {
            suma -= L[i * numDimensiones + j] * v[j];
        }

        v[i] = suma / L[i * numDimensiones + i];
    }
}

double GMM::logaritmoDeterminante() // logarithmDeterminant 协方差矩阵Sigma行列式对数
{
    double suma = 0.0;

    for (size_t i = 0; i < numDimensiones; i++) {
        suma += gsl_sf_log(L[i * numDimensiones + i]); // Cholesky分解协方差矩阵Sigma得到的L矩阵对角线上元素的对数求和, 因为行列式等于对角线元素之积
    }

    return 2.0 * suma; // Sigma的行列式等于L行列式的平方
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
