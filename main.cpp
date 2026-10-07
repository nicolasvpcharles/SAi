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

    //=======================================
    // Fontion d entrainement
    //=======================================
    void train(int eproach = 1000, double learningRate = 0.01)
    {
        // on entraine ici le reseaux de neurone
        // il faut donner un weight aleatoire pour commencer l entrainement
        bias = randomWeight();

        weight.clear();

        for (int i = 0; i < inputLayer[0].size(); i++)
        {
            weight.push_back(randomWeight());
        };

        // normalement on a des weight alleatoires
        // maintenant il va faloir lire une documentation pour comprendre comment le tout marche
        // resources
        // https://www.freecodecamp.org/news/neural-networks-explained-simply-in-python/
        //

        // ici maintenant on fait une boucle while et on fix apres le bail
        // pour trouver une reponse on doit faire le calcul suivant :
        // z = x₁w₁ + x₂w₂ + x₃w₃ + b
        //
        int i = 0;

        while (i <= eproach)
        {
            // on aplique le calcul
            // attention que il faut bien faire la bail pour chaque x et w
            int a = 0;

            while (a < inputLayer.size())
            {
                // il faut faire une 3eme boucle mtn
                int n = 0;

                // on commence avec le bias
                double prediction = bias;

                while (n < inputLayer[a].size())
                {
                    // maintenant dans le bail je vais
                    prediction = prediction + (inputLayer[a][n] * weight[n]);

                    n = n + 1;
                };

                // on calcule l'erreur entre la prediction et la bonne reponse
                double error = prediction - awnsers[a];

                // c est ici que on doit corriger les weight et bias

                n = 0;

                while (n < inputLayer[a].size())
                {
                    weight[n] = weight[n] - learningRate * error * inputLayer[a][n];

                    n = n + 1;
                };

                // on corrige le bias
                bias = bias - learningRate * error;

                a = a + 1;
            };

            i = i + 1;
        };
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
    network.train();
    cout << "bias" << endl;
    cout << network.bias << endl;
    cout << "weights" << endl;
    for (int i = 0; i < network.weight.size(); i++)
    {
        cout << network.weight[i] << endl;
    }
    return 0;
};
