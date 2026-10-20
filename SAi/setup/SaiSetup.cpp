#include <iostream>
#include "../neuralNetwork.h"
#include "../modules/datasetopen.h"
#include <vector>
#include <string>
using namespace std;
int main()
{
    bool running = true;
    while (true)
    {

        int choice;
        // ici ça va etre une interface terminal pour setup le nn plus facilement
        cout << "Sai setup" << endl;
        cout << "1. train a model\n2. calibrate a model\n3.evaluate a model\n5. quitt" << endl
             << "/";
        cin >> choice;
        // il faut que j ajoute une maniere de load les marametres du modele
        if (choice == 1)
        {
            string trainingDatasetDirectory;
            string resultDatasetDirectory;
            int nInputNeurons;
            int nHidenNeurons;
            // train a model
            // here we will type the important data such as the networks input neurons idden neurons etc
            cout << endl
                 << "write the directory of the dataset training dataset" << endl
                 << "/";
            cin >> trainingDatasetDirectory;
            vector<vector<double>> trainingDataset = loadDatasetCsv(trainingDatasetDirectory);
            // resultDataset
            cout << endl
                 << "write the directory of the dataset result dataset" << endl
                 << "/";
            cin >> resultDatasetDirectory;
            vector<double> resultDataset = loadResultCsv(trainingDatasetDirectory);
            //
            // neurons dataset
            //
            // en theorie le nombre de neurones input c est e nombre de data donc a la limite je fait juste un bailou je calcule la len de input directory
            // je dois juste faire verifier la size des 2 datasets
            if (resultDataset.size() != trainingDataset.size())
            {
                cout << "erreur les datasets ont une taille differente";
            }
            else
            {
                // ici le programme continue
                nInputNeurons = (trainingDataset[0]).size();
                cout << endl
                     << "write the number of hidden neurons" << endl
                     << "/";
                cin >> nHidenNeurons;
                // maintenant on peut setup le neural network
                neuralNetwork nn(trainingDataset, resultDataset, nInputNeurons, nHidenNeurons);
                // maintenant il faut que je fasse une option qui train avec un nombre de eproachs
                int eproach = -1;
                while (eproach <= 0)
                {
                    cout << endl
                         << "write the numbre of eproach for your model to train" << endl
                         << "/";
                    cin >> eproach;
                };
                nn.train(eproach);
                cout << "saving the model to /neuralNetwork";
                nn.save("neuralNetwork");
            };
        };
        if (choice == 2)
        {
            // calibrate a model
        }
        if (choice == 3)
        {
            // evaluate a model
        }
        if (choice == 5)
        {

            running = false;
        }

        else
        {
            cout << "there is no choice " << choice << " .";
        }
    };
};