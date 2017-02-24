#include <cmath>
#include <random>
#include <chrono>
#include <fstream>
#include <iostream>

#include "gmm.h"

using namespace std;
using namespace std::chrono;

// 在指定范围内产生随机数
unsigned int numeroAleatorio(unsigned int rango) // numero=number Aleatorio=random
{
    random_device randomDevice;
    mt19937 generador(randomDevice());
    uniform_int_distribution<unsigned int> uniformDistribution(0, rango - 1);

    return uniformDistribution(generador);
}

int main(int argc, char* argv[])
{
    // Abrimos el fichero que nos pasan como argumento en modo lectura // fichero=file, 打开读入文件
    ifstream fichero(argv[1]);

    if (!fichero.is_open()) {
        cerr << "ERROR: no se pudo abrir el fichero.\n";
        return -1;
    }

    // Abrimos el segundo fichero que nos pasan como argumento en modo escritura // 打开写出文件
    ofstream ficheroOutput(argv[2]);

    if (!ficheroOutput.is_open()) {
        cerr << "ERROR: no se pudo abrir el fichero de salida.\n";
        return -1;
    }

    size_t numMuestras = 0; // Muestra=Sample 样本数
    size_t numDimensiones = 0; // 维数
    size_t numPuntos = 0; // Punto=point 点数, 也就是输入矩阵的元素个数

    // Leemos la cabecera // 读取文件的第一行
    double basura;
    fichero >> numMuestras; // Muestra=Sample 样本数
    fichero >> numDimensiones; // 维数
    fichero >> basura; // basura=trash

    numPuntos = numMuestras * numDimensiones; // 点数=样本数*维数, 也就是输入矩阵的元素个数

    double *datos = new double[numPuntos];

    // Cargamos los datos en memoria // 将数据载入内存, 注意这里是column-major的，方便在CUDA中使用(CPU版是row-major的)
    // datos矩阵形式为
    // sample11 sample12 ...
    // sample21 sample22 ...
    for (size_t i = 0; i < numMuestras; i++) {
        for (size_t j = 0; j < numDimensiones; j++) {
            fichero >> datos[j * numMuestras + i];
        }
    }

    // Cerramos el fichero // 关闭文件
    fichero.close();

    // Definimos algunos parametros importantes // 定义一些重要的参数
    const unsigned int maxIteraciones = 500; // 最大迭代次数
    const double precision = 1.0e-5; // 停止精度

    double *media_muestras = new double[numDimensiones]; // 每个mixture的均值向量
    double *matriz_covarianza = new double[numDimensiones * numDimensiones]; // 每个mixture的协方差矩阵

    double suma;

    // Calculamos el vector media de todas las muestras y la matriz de covarianza
    // 计算整个数据集的均值和协方差矩阵
    // 整个数据集的均值
    for (size_t j = 0; j < numDimensiones; j++) {
        suma = 0.0;

        for (size_t i = 0; i < numMuestras; i++) {
            suma += datos[j * numMuestras + i];
        }

        media_muestras[j] = suma / numMuestras;
    }

    // 整个数据集的协方差矩阵
    for (size_t j = 0; j < numDimensiones; j++) {
        for (size_t h = j; h < numDimensiones; h++) {
            suma = 0.0;

            for (size_t i = 0; i < numMuestras; i++) {
                suma += (datos[j * numMuestras + i] - media_muestras[j])
                    * (datos[h * numMuestras + i] - media_muestras[h]);
            }

            matriz_covarianza[j * numDimensiones + h] = suma / (numMuestras - 1); // 注意这里是unbiased的
            matriz_covarianza[h * numDimensiones + j] = matriz_covarianza[j * numDimensiones + h];
        }
    }

    // Definimos la lista de numeros de gaussianas que queremos comprobar
    // 定义不同GMM的数量, 通过遍历来找到最好的mixture个数
    size_t numGaussianasLista[] = { 1, 2, 5, 10, 15, 20 };

    // Iteramos entre los distintos numeros de gaussianas que queremos comprobar
    // 在不同GMM数量间进行迭代
    for (size_t z = 0; z < sizeof(numGaussianasLista) / sizeof(*numGaussianasLista); z++) {
        size_t numGaussianas = numGaussianasLista[z]; // GMM数量

        double *pesos = new double[numGaussianas]; // peso=weight 权重
        double *medias = new double[numGaussianas * numDimensiones]; // 各个mixture的均值向量
        double *covarianzas = new double[numGaussianas * numDimensiones * numDimensiones]; // 各个mixture的协方差矩阵

        // Definimos los parametros iniciales // 定义初始参数
        for (size_t k = 0; k < numGaussianas; k++) {
            pesos[k] = 1.0 / numGaussianas; // 权重初始化

            // Generamos datos aleatorios para las medias
            // 随机产生各个mixture的均值向量和协方差矩阵(协方差矩阵初始化为上面计算得到的整个数据集的协方差矩阵)
            size_t muestra = (size_t) numeroAleatorio((unsigned int) numMuestras);

            for (size_t i = 0; i < numDimensiones; i++) {
                medias[k * numDimensiones + i] = datos[i * numMuestras + muestra];

                for (size_t j = 0; j < numDimensiones; j++) {
                    covarianzas[k * numDimensiones * numDimensiones + i * numDimensiones + j] = matriz_covarianza[i * numDimensiones + j];
                }
            }
        }

        cerr << "****************************************\n";
        cerr << "Ejecutando algoritmo usando " << numGaussianas << " gaussianas"; // 使用numGaussianas各高斯执行算法

        // Medimos el tiempo inicial // 开始时间
        high_resolution_clock::time_point tInicio = high_resolution_clock::now();

        // Creamos el modelo y lo ejecutamos // We create the model and execute it
        GMM gmm(numGaussianas, pesos, medias, covarianzas, maxIteraciones, precision);

        auto resultados = gmm.estimar(datos, numMuestras, numDimensiones);

        // Medimos el tiempo final // 终止时间
        high_resolution_clock::time_point tFin = high_resolution_clock::now();

        auto tiempo = duration_cast<milliseconds>(tFin - tInicio).count();

        cerr << "Tiempo de ejecucion: " << tiempo << " ms." << endl << endl; // 执行时间

        ficheroOutput << numGaussianas << "," << tiempo << ","
            << resultados.iteracion << "," << resultados.verosimilitud << endl;

        delete [] pesos; // peso=weight 权重
        delete [] medias; // 均值
        delete [] covarianzas; // 协方差矩阵
    }

    // Cerramos el fichero de salida // 关闭文件
    ficheroOutput.close();

    // Limpiamos la memoria // 释放内存
    delete [] datos;
    delete [] media_muestras;
    delete [] matriz_covarianza;

    return 0;
}
