#include <cmath>
#include <random>
#include <chrono>
#include <fstream>
#include <iostream>

#include "gmm.h"

using namespace std;
using namespace std::chrono;

unsigned int numeroAleatorio(unsigned int rango)
{
    random_device randomDevice;
    mt19937 generador(randomDevice());
    uniform_int_distribution<unsigned int> uniformDistribution(0, rango - 1);

    return uniformDistribution(generador);
}

int main(int argc, char* argv[])
{
    // Abrimos el fichero que nos pasan como argumento en modo lectura
    ifstream fichero(argv[1]);

    if (!fichero.is_open()) {
        cerr << "ERROR: no se pudo abrir el fichero.\n";
        return -1;
    }

    // Abrimos el segundo fichero que nos pasan como argumento en modo escritura
    ofstream ficheroOutput(argv[2]);

    if (!ficheroOutput.is_open()) {
        cerr << "ERROR: no se pudo abrir el fichero de salida.\n";
        return -1;
    }

    size_t numMuestras = 0;
    size_t numDimensiones = 0;
    size_t numPuntos = 0;

    // Leemos la cabecera
    double basura;
    fichero >> numMuestras;
    fichero >> numDimensiones;
    fichero >> basura;

    numPuntos = numMuestras * numDimensiones;

    double *datos = new double[numPuntos];

    // Cargamos los datos en memoria
    for (size_t i = 0; i < numPuntos; i++) {
        fichero >> datos[i];
    }

    // Cerramos el fichero
    fichero.close();

    // Definimos algunos parametros importantes
    const unsigned int maxIteraciones = 500;
    const double precision = 1.0e-5;

    double *media_muestras = new double[numDimensiones];
    double *matriz_covarianza = new double[numDimensiones * numDimensiones];

    double suma;

    // Calculamos el vector media de todas las muestras y la matriz de covarianza
    for (size_t j = 0; j < numDimensiones; j++) {
        suma = 0.0;

        for (size_t i = 0; i < numMuestras; i++) {
            suma += datos[i * numDimensiones + j];
        }

        media_muestras[j] = suma / numMuestras;
    }

    for (size_t j = 0; j < numDimensiones; j++) {
        for (size_t h = j; h < numDimensiones; h++) {
            suma = 0.0;

            for (size_t i = 0; i < numMuestras; i++) {
                suma += (datos[i * numDimensiones + j] - media_muestras[j])
                    * (datos[i * numDimensiones + h] - media_muestras[h]);
            }

            matriz_covarianza[j * numDimensiones + h] = suma / (numMuestras - 1);
            matriz_covarianza[h * numDimensiones + j] = matriz_covarianza[j * numDimensiones + h];
        }
    }

    // Definimos la lista de numeros de gaussianas que queremos comprobar
    size_t numGaussianasLista[] = { 1, 2, 5, 10, 15, 20 };

    // Iteramos entre los distintos numeros de gaussianas que queremos comprobar
    for (size_t z = 0; z < sizeof(numGaussianasLista) / sizeof(*numGaussianasLista); z++) {
        size_t numGaussianas = numGaussianasLista[z];

        double *pesos = new double[numGaussianas];
        double *medias = new double[numGaussianas * numDimensiones];
        double *covarianzas = new double[numGaussianas * numDimensiones * numDimensiones];

        // Definimos los parametros iniciales
        for (size_t k = 0; k < numGaussianas; k++) {
            pesos[k] = 1.0 / numGaussianas;

            // Generamos datos aleatorios para las medias
            size_t muestra = (size_t) numeroAleatorio((unsigned int) numMuestras);

            for (size_t i = 0; i < numDimensiones; i++) {
                medias[k * numDimensiones + i] = datos[muestra * numDimensiones + i];

                for (size_t j = 0; j < numDimensiones; j++) {
                    covarianzas[k * numDimensiones * numDimensiones + i * numDimensiones + j] = matriz_covarianza[i * numDimensiones + j];
                }
            }
        }

        cerr << "****************************************\n";
        cerr << "Ejecutando algoritmo usando " << numGaussianas << " gaussianas";

        // Medimos el tiempo inicial
        high_resolution_clock::time_point tInicio = high_resolution_clock::now();

        // Creamos el modelo y lo ejecutamos
        GMM gmm(numGaussianas, pesos, medias, covarianzas, maxIteraciones, precision);

        auto resultados = gmm.estimar(datos, numMuestras, numDimensiones);

        // Medimos el tiempo final
        high_resolution_clock::time_point tFin = high_resolution_clock::now();

        auto tiempo = duration_cast<milliseconds>(tFin - tInicio).count();

        cerr << "Tiempo de ejecucion: " << tiempo << " ms." << endl << endl;

        ficheroOutput << numGaussianas << "," << tiempo << ","
            << resultados.iteracion << "," << resultados.verosimilitud << endl;

        delete [] pesos;
        delete [] medias;
        delete [] covarianzas;
    }

    // Cerramos el fichero de salida
    ficheroOutput.close();

    // Limpiamos la memoria
    delete [] datos;
    delete [] media_muestras;
    delete [] matriz_covarianza;

    return 0;
}
