#include <iostream>
#include <vector>
#include <cmath>
#include <random>
using namespace std;

// for calculating a neural network it is n1 + w1 + b
class neuralNetwork
{
public:
    vector<vector<double>> inputLayer;
    vector<double> awnsers;
    vector<vector<double>> weight;
    vector<double> bias;

    // poids entre la couche cachee et la sortie
    vector<double> outputWeight;

    // bias de la sortie
    double outputBias;

    double outputLayer;
    double nNeurons;
    double neuronDepth;

    // fonction relu
    neuralNetwork(vector<vector<double>> nInputlayer, vector<double> nAwnsers, double numberNeurons, double nNeuronDepth)
    {
        inputLayer = nInputlayer;
        awnsers = nAwnsers;
        nNeurons = numberNeurons;
        neuronDepth = nNeuronDepth;
    };

    double randomWeight()
    {
        static random_device rd;
        static mt19937 gen(rd());
        static uniform_real_distribution<double> dist(-1.0, 1.0);

        return dist(gen);
    }

    double ReLU(double n)
    {
        if (n < 0.0)
        {
            return 0.0;
        }
        else
        {
            return n;
        };
    };

    //====================================
    // calculate loss
    //====================================
    double loss(double prediction, double awnser)
    {

        return ((prediction - awnser) * (prediction - awnser));
    };
    //==================================
    // Gradient descent
    //====================================
    double gradientDescent(double weight, double learningRate, double gradient)
    {
        weight = weight - learningRate * gradient;
        return weight;
    };

    //=====================================
    // Multiplication de matrices
    //=====================================
    vector<vector<double>> multiplyMatrices(
        vector<vector<double>> A,
        vector<vector<double>> B)
    {
        vector<vector<double>> result(
            A.size(),
            vector<double>(B[0].size(), 0));

        for (int i = 0; i < A.size(); i++)
        {
            for (int j = 0; j < B[0].size(); j++)
            {
                for (int k = 0; k < B.size(); k++)
                {
                    result[i][j] += A[i][k] * B[k][j];
                }
            }
        }

        return result;
    };
    //
    // FONCTION SIGMOIDE
    //
    double sigmoid(double x)
    {
        return 1.0 / (1.0 + exp(-x));
    };

    //
    // FONCTION TANH
    //
    double tanhActivation(double x)
    {
        return tanh(x);
    };

    //
    // DERIVE TANH
    //
    double tanhDerivative(double x)
    {
        return 1.0 - x * x;
    };

    //
    // DERIVE SIGMOIDE
    //
    double sigmoidDerivative(double x)
    {
        return x * (1.0 - x);
    };

    //=======================================
    // Fontion d entrainement
    //=======================================
    void train(int eproach = 1000, double learningRate = 0.01)
    {
        // on entraine ici le reseaux de neurone
        // il faut donner un weight aleatoire pour commencer l entrainement

        bias.clear();
        weight.clear();
        outputWeight.clear();

        // on cree les neurones
        for (int i = 0; i < nNeurons; i++)
        {
            vector<double> neuronWeights;

            // chaque neurone a un weight pour chaque input
            for (int j = 0; j < inputLayer[0].size(); j++)
            {
                neuronWeights.push_back(randomWeight());
            }

            weight.push_back(neuronWeights);
            bias.push_back(randomWeight());

            // poids entre le neurone cache et la sortie
            outputWeight.push_back(randomWeight());
        }

        // bias de la sortie
        outputBias = randomWeight();

        // normalement on a des weight alleatoires
        // maintenant il va faloir lire une documentation pour comprendre comment le tout marche
        // resources
        // https://www.freecodecamp.org/news/neural-networks-explained-simply-in-python/

        int i = 0;

        while (i < eproach)
        {
            int a = 0;

            while (a < inputLayer.size())
            {
                // on parcourt chaque neurone
                int neuron = 0;

                //========================================
                // PREMIERE COUCHE
                //========================================

                vector<double> hiddenOutput;

                while (neuron < nNeurons)
                {
                    // on commence avec le bias du neurone
                    double prediction = bias[neuron];

                    // on calcule la prediction
                    int n = 0;

                    while (n < inputLayer[a].size())
                    {
                        prediction = prediction +
                                     (inputLayer[a][n] * weight[neuron][n]);

                        n = n + 1;
                    }

                    // fonction d activation tanh
                    prediction = tanhActivation(prediction);

                    hiddenOutput.push_back(prediction);

                    neuron = neuron + 1;
                }

                //========================================
                // DEUXIEME COUCHE
                //========================================

                double output = outputBias;

                neuron = 0;

                while (neuron < nNeurons)
                {
                    output = output +
                             (hiddenOutput[neuron] * outputWeight[neuron]);

                    neuron = neuron + 1;
                }

                // fonction sigmoid pour avoir une prediction entre 0 et 1
                double prediction = sigmoid(output);

                // on calcule l'erreur
                double error = prediction - awnsers[a];

                //========================================
                // BACKPROPAGATION
                //========================================

                // gradient de la sortie
                double outputGradient =
                    error * sigmoidDerivative(prediction);

                // correction des poids de sortie
                neuron = 0;

                while (neuron < nNeurons)
                {
                    double gradient =
                        outputGradient * hiddenOutput[neuron];

                    outputWeight[neuron] =
                        outputWeight[neuron] -
                        learningRate * gradient;

                    neuron = neuron + 1;
                }

                // correction du bias de sortie
                outputBias =
                    outputBias -
                    learningRate * outputGradient;

                //========================================
                // CORRECTION COUCHE CACHEE
                //========================================

                neuron = 0;

                while (neuron < nNeurons)
                {
                    double hiddenGradient =
                        outputGradient *
                        outputWeight[neuron] *
                        tanhDerivative(hiddenOutput[neuron]);

                    // on corrige les weights
                    int n = 0;

                    while (n < inputLayer[a].size())
                    {
                        weight[neuron][n] =
                            weight[neuron][n] -
                            learningRate *
                                hiddenGradient *
                                inputLayer[a][n];

                        n = n + 1;
                    }

                    // on corrige le bias
                    bias[neuron] =
                        bias[neuron] -
                        learningRate * hiddenGradient;

                    neuron = neuron + 1;
                }

                a = a + 1;
            }

            i = i + 1;
        }
    };

    void predict(vector<vector<double>> input)
    {
        // en utilisant les weights on peut predire un output
        // il faut que je relise la doc le bail est dur smr

        int a = 0;

        while (a < input.size())
        {
            vector<double> hiddenOutput;

            // on parcourt les neurones de la couche cachee
            int neuron = 0;

            while (neuron < nNeurons)
            {
                double prediction = bias[neuron];

                int n = 0;

                while (n < input[a].size())
                {
                    prediction =
                        prediction +
                        (input[a][n] * weight[neuron][n]);

                    n = n + 1;
                }

                // activation tanh
                prediction = tanhActivation(prediction);

                hiddenOutput.push_back(prediction);

                neuron = neuron + 1;
            }

            //========================================
            // SORTIE
            //========================================

            double output = outputBias;

            neuron = 0;

            while (neuron < nNeurons)
            {
                output =
                    output +
                    (hiddenOutput[neuron] *
                     outputWeight[neuron]);

                neuron = neuron + 1;
            }

            // sigmoid
            double prediction = sigmoid(output);

            cout << "Input : ";

            int n = 0;

            while (n < input[a].size())
            {
                cout << input[a][n] << " ";
                n = n + 1;
            }

            cout << "-> Prediction : "
                 << prediction;

            if (prediction >= 0.5)
            {
                cout << " -> 1";
            }
            else
            {
                cout << " -> 0";
            }

            cout << endl;

            a = a + 1;
        }
    };
};

int main()
{

    // test scene
    vector<vector<double>> q = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}};
    vector<double> r = {0.0, 1.0, 1.0, 0.0};
    neuralNetwork network(q, r, 2, 1);
    network.train(10000, 0.1);
    cout << "bias" << endl;
    for (int i = 0; i < network.bias.size(); i++)
    {
        cout << "Bias " << i << " : " << network.bias[i] << endl;
    }

    cout << "weights" << endl;
    for (int i = 0; i < network.weight.size(); i++)
    {
        cout << "Neuron " << i << " : ";

        for (int j = 0; j < network.weight[i].size(); j++)
        {
            cout << network.weight[i][j] << " ";
        }

        cout << endl;
    }

    // il faut que mtn que j ajoute une prediction pour le bail

    cout << endl;
    cout << "Output weights : ";

    for (int i = 0; i < network.outputWeight.size(); i++)
    {
        cout << network.outputWeight[i] << " ";
    }

    cout << endl;

    cout << "Output bias : "
         << network.outputBias
         << endl;

    cout << endl;
    cout << "Predictions :" << endl;

    network.predict(q);

    return 0;
};
