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

    delete [] this->resp;
    delete [] this->L;
    delete [] this->v;
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

    this->resp = new double[numMuestras * numGaussianas];
    this->L = new double[numDimensiones * numDimensiones];
    this->v = new double[numDimensiones];

    this->datos = datos;
    this->numMuestras = numMuestras;
    this->numDimensiones = numDimensiones;

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
    double *logDets = new double[numGaussianas];
    double *u = new double[numDimensiones];
    double suma, verosimilitudParcial, maxContribucion, sumaProbabilidades;

    // Paso E
    for (size_t k = 0; k < numGaussianas; k++) {
        cholesky(k);
        logDets[k] = logaritmoDeterminante();

        for (size_t i = 0; i < numMuestras; i++) {
            for (size_t j = 0; j < numDimensiones; j++) {
                u[j] = datos[i * numDimensiones + j] - medias[k * numDimensiones + j];
            }

            resolver(u);

            suma = 0.0;

            for (size_t j = 0; j < numDimensiones; j++) {
                suma += v[j] * v[j];
            }

            resp[i * numGaussianas + k] = -0.5 * (suma + logDets[k]) + gsl_sf_log(pesos[k]);
        }
    }

    verosimilitud = 0.0;

    for (size_t i = 0; i < numMuestras; i++) {
        maxContribucion = MENOS_INFINITO;

        for (size_t k = 0; k < numGaussianas; k++) {
            if (resp[i * numGaussianas + k] > maxContribucion) {
                maxContribucion = resp[i * numGaussianas + k];
            }
        }

        suma = 0.0;

        for (size_t k = 0; k < numGaussianas; k++) {
            suma += exp(resp[i * numGaussianas + k] - maxContribucion);
        }

        verosimilitudParcial = maxContribucion + gsl_sf_log(suma);

        for (size_t k = 0; k < numGaussianas; k++) {
            resp[i * numGaussianas + k] = exp(resp[i * numGaussianas + k] - verosimilitudParcial);
        }

        verosimilitud += verosimilitudParcial;
    }

    // Paso M
    for (size_t k = 0; k < numGaussianas; k++) {
        sumaProbabilidades = 0.0;

        for (size_t i = 0; i < numMuestras; i++) {
            sumaProbabilidades += resp[i * numGaussianas + k];
        }

        pesos[k] = sumaProbabilidades / numMuestras;

        for (size_t j = 0; j < numDimensiones; j++) {
            suma = 0.0;

            for (size_t i = 0; i < numMuestras; i++) {
                suma += resp[i * numGaussianas + k] * datos[i * numDimensiones + j];
            }

            medias[k * numDimensiones + j] = suma / sumaProbabilidades;

            for (size_t h = 0; h < numDimensiones; h++) {
                suma = 0.0;

                for (size_t i = 0; i < numMuestras; i++) {
                    suma += resp[i * numGaussianas + k] * (datos[i * numDimensiones + j] - medias[k * numDimensiones + j])
                        * (datos[i * numDimensiones + h] - medias[k * numDimensiones + h]);
                }

                covarianzas[k * numDimensiones * numDimensiones + j * numDimensiones + h] = suma / sumaProbabilidades;
            }
        }
    }

    // Limpiamos la memoria
    delete [] logDets;
    delete [] u;
}

void GMM::cholesky(size_t gaussiana)
{
    for (size_t i = 0; i < numDimensiones; i++) {
        for (size_t j = 0; j < numDimensiones; j++) {
            L[i * numDimensiones + j] = 0.0;
        }
    }

    for (size_t i = 0; i < numDimensiones; i++) {
        for (size_t j = 0; j < i + 1; j++) {
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

void GMM::resolver(double *muestra)
{
    double suma;

    for (size_t i = 0; i < numDimensiones; i++) {
        v[i] = 0.0;
    }

    for (size_t i = 0; i < numDimensiones; i++) {
        suma = muestra[i];

        for (size_t j = 0; j < i; j++) {
            suma -= L[i * numDimensiones + j] * v[j];
        }

        v[i] = suma / L[i * numDimensiones + i];
    }
}

double GMM::logaritmoDeterminante()
{
    double suma = 0.0;

    for (size_t i = 0; i < numDimensiones; i++) {
        suma += gsl_sf_log(L[i * numDimensiones + i]);
    }

    return 2.0 * suma;
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
