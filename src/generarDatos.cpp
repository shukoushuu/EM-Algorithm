#include <random>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace std;

double numeroAleatorio(int centro, int rango)
{
    random_device randomDevice;
    mt19937 generador(randomDevice());
    uniform_real_distribution<double> uniformDistribution(centro - rango, centro + rango);

    return uniformDistribution(generador);
}

int main(int argc, char* argv[])
{
    ofstream fichero(argv[1]);

    if (!fichero.is_open()) {
        cerr << "ERROR: no se pudo abrir el fichero.\n";
        return -1;
    }

    int numGaussianas = atoi(argv[2]);
    int numMuestras = atoi(argv[3]);
    int numDimensiones = atoi(argv[4]);

    int numPuntos = numMuestras * numDimensiones;

    int *centros = new int[numGaussianas];

    for (int i = 0; i < numGaussianas; i++) {
        centros[i] = 5 + i * 10;
    }

    fichero << setprecision(6);

    fichero << numMuestras << " ";
    fichero << numDimensiones << " ";
    fichero << "0" << endl;

    for (int i = 0; i < numMuestras; i++) {
        for (int j = 0; j < numDimensiones; j++) {
            fichero << numeroAleatorio(centros[i % numGaussianas], 5);

            if (j != numDimensiones - 1) {
                 fichero << " ";
            }
        }

        fichero << endl;
    }

    fichero.close();

    return 0;
}
