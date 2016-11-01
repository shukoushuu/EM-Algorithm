#ifndef GMM_H
#define GMM_H

struct resultado_type {
  size_t iteracion;
  double verosimilitud;
};

class GMM
{
  public:

    GMM(size_t numGaussianas, double* pesos, double* medias, double* covarianzas, unsigned int maxIteraciones, double precision);
    ~GMM();

    resultado_type estimar(double *datos, size_t numMuestras, size_t numDimensiones);

  private:

    double *datos;
    double *pesos;
    double *medias;
    double *covarianzas;

    size_t numMuestras;
    size_t numGaussianas;
    size_t numDimensiones;
    size_t numRespuestasGPU;

    unsigned int maxIteraciones;
    double precision;

    double verosimilitud;

    void calcular();

    void inicializarGPU();
    void copiarEnGPU();
    void copiarDesdeGPU();
    void limpiarGPU();

    void imprimir(bool finalizado, unsigned int iteracion);
};

#endif
