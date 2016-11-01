#ifndef GMM_H
#define GMM_H

#include <limits>

using namespace std;

// Constantes
const double MENOS_INFINITO = -numeric_limits<double>::max();

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

    double *pesos;
    double *medias;
    double *covarianzas;

    double *resp;
    double *L;
    double *v;

    size_t numMuestras;
    size_t numGaussianas;
    size_t numDimensiones;
    double *datos;

    unsigned int maxIteraciones;
    double precision;

    double verosimilitud;

    void calcular();

    void cholesky(size_t gaussiana);
    void resolver(double *vector);
    double logaritmoDeterminante();

    void imprimir(bool finalizado, unsigned int iteracion);
};

#endif
