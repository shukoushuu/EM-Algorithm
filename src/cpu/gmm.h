#ifndef GMM_H
#define GMM_H

#include <limits>

using namespace std;

// Constantes
const double MENOS_INFINITO = -numeric_limits<double>::max(); // MENOS=minimum 负无穷

struct resultado_type { // resultado = result
  size_t iteracion; // iteracion = iteration
  double verosimilitud; // verosimilitud=likelihood 似然
};

class GMM
{
  public:

    GMM(size_t numGaussianas, double* pesos, double* medias, double* covarianzas, unsigned int maxIteraciones, double precision);
    ~GMM();

    resultado_type estimar(double *datos, size_t numMuestras, size_t numDimensiones); // estimar=estimate

  private:

    double *pesos; // peso=weight 各mixture权重
    double *medias; // 各mixture均值
    double *covarianzas; // 各mixture协方差矩阵

    double *resp; // responsibility矩阵, 存储p_nk, 也就是p(k|n), 第n个点属于第k个mixture的概率
    double *L; // cholesky分解协方差矩阵获得的下三角矩阵L L*L'为协方差矩阵
    double *v;

    size_t numMuestras; // 待分类的样本数量
    size_t numGaussianas; // 高斯mixture数量
    size_t numDimensiones; // 维数
    double *datos; // 输入的数据

    unsigned int maxIteraciones; // 最大迭代次数
    double precision; // 停止精度

    double verosimilitud; // 似然

    void calcular(); // calcular=calculate, 

    void cholesky(size_t gaussiana);
    void resolver(double *vector);
    double logaritmoDeterminante();

    void imprimir(bool finalizado, unsigned int iteracion); // imprimir=print输出
};

#endif
