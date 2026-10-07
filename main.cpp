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
    vector<double> weight;
    double bias;

    double outputLayer;
    double nNeurons;
    double neuronDepth;

    //=======================================
    // CONSTRUCTEUR
    //=======================================
    neuralNetwork(
        vector<vector<double>> nInputlayer,
        vector<double> nAwnsers,
        double numberNeurons,
        double nNeuronDepth)
    {
        inputLayer = nInputlayer;
        awnsers = nAwnsers;
        nNeurons = numberNeurons;
        neuronDepth = nNeuronDepth;
    };

    //=======================================
    // POIDS ALEATOIRE
    //=======================================
    double randomWeight()
    {
        static random_device rd;
        static mt19937 gen(rd());
        static uniform_real_distribution<double> dist(-1.0, 1.0);

        return dist(gen);
    }

    //=======================================
    // FONCTION RELU
    //=======================================
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
    // CALCULATE LOSS
    //====================================
    double loss(double prediction, double awnser)
    {
        return ((prediction - awnser) * (prediction - awnser));
    };

    //==================================
    // GRADIENT DESCENT
    //====================================
    double gradientDescent(
        double weight,
        double learningRate,
        double gradient)
    {
        weight = weight - learningRate * gradient;

        return weight;
    };

    //=====================================
    // MULTIPLICATION DE MATRICES
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

    //=======================================
    // FONCTION SIGMOIDE
    //=======================================
    double sigmoid(double x)
    {
        return 1.0 / (1.0 + exp(-x));
    };

    //=======================================
    // FORWARD PROPAGATION
    //=======================================
    double forwardPropagation(vector<double> input)
    {
        double result = 0.0;

        // weight × input
        for (int i = 0; i < input.size(); i++)
        {
            result += input[i] * weight[i];
        }

        // + bias
        result += bias;

        // activation
        result = sigmoid(result);

        return result;
    };

    //=======================================
    // FONCTION D'ENTRAINEMENT
    //=======================================
    double train(int eproach = 1000)
    {
        // Donner un bias aleatoire
        bias = randomWeight();

        // Donner des poids aleatoires
        weight.clear();

        for (int i = 0; i < inputLayer[0].size(); i++)
        {
            weight.push_back(randomWeight());
        }

        // Tester le forward propagation
        for (int i = 0; i < inputLayer.size(); i++)
        {
            double prediction =
                forwardPropagation(inputLayer[i]);

            cout << "Input : ";

            for (double value : inputLayer[i])
            {
                cout << value << " ";
            }

            cout << endl;

            cout << "Prediction : "
                 << prediction << endl;

            cout << "Answer : "
                 << awnsers[i] << endl;

            cout << "Loss : "
                 << loss(prediction, awnsers[i])
                 << endl;

            cout << "----------------------"
                 << endl;
        }

        return 0;
    };
};

int main()
{
    //=======================================
    // XOR DATASET
    //=======================================

    vector<vector<double>> q = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}};

    vector<double> r = {
        0.0,
        1.0,
        1.0,
        0.0};

    //=======================================
    // CREATION DU RESEAU
    //=======================================

    neuralNetwork network(q, r, 2, 1);

    //=======================================
    // ENTRAINEMENT
    //=======================================

    network.train();

    return 0;
};