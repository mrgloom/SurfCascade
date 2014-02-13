#include "LearningAlgorithms/CascadeClassifier/WeakClassifiers/LogisticRegression.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <string>

using namespace std;

int main()
{
    ifstream f("C:/Users/Meng/Documents/weekly.csv");
    //ifstream f("C:/Users/Meng/Documents/default1.csv");
    //ifstream f("C:/Users/Meng/Documents/smarket.csv");
    string line;
    vector<vector<double>> X;
    vector<bool> y;
    double a;

    while (getline(f, line))
    {
        X.push_back(vector<double>());
        stringstream ss(line);
        while (ss >> a)
        {
            if (ss.peek() == ',')
            {
                X.back().push_back(a);
                ss.ignore();
            }
            else
                y.push_back((bool)a);
        }
    }

    LogisticRegression lr(0);

    lr.Train(X, y);
    lr.Print();

    return 0;
}