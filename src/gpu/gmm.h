#ifndef GMM_H
#define GMM_H

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

    double *datos; // 输入的数据
    double *pesos; // peso=weight 各mixture权重
    double *medias; // 各mixture均值
    double *covarianzas; // 各mixture协方差矩阵

    size_t numMuestras; // 待分类的样本数量
    size_t numGaussianas; // 高斯mixture数量
    size_t numDimensiones; // 维数
    size_t numRespuestasGPU; // 需要的BLOCK数量

    unsigned int maxIteraciones; // 最大迭代次数
    double precision; // 停止精度

    double verosimilitud; // 似然

    void calcular(); // calcular=calculate, 

    void inicializarGPU(); // inicializar=initialize
    void copiarEnGPU(); // copy to GPU
    void copiarDesdeGPU(); // copy from GPU 从GPU拷贝到CPU
    void limpiarGPU(); // limpiar=clean, 释放GPU上分配的内存

    void imprimir(bool finalizado, unsigned int iteracion); // imprimir=print输出
};

#endif
