#include <iostream>
#include <vector>
#include "neuralNetwork.h"

using namespace std;

int main()
{
    vector<vector<double>> inputs = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}};

    vector<double> answers = {
        0,
        1,
        1,
        0};

    neuralNetwork nn(inputs, answers, 4, 1);

    nn.train(10000, 0.1);

    nn.predict(inputs);

    return 0;
}
