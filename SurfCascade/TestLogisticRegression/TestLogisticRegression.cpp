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
    //vector<vector<double>> X = { { 1, 1 }, { 2, 1 }, { 1, 6 }, { 3, 4 }, { 5, 2 }, { 7, 9 }, { 8, 3 }, { 1.5, 6 }, { 10, 11 } };
    //vector<double> y = { 3, 4, 13, 11, 9, 25, 14, 13.5, 32 };
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
                y.push_back(!!a); // convert to bool without warning
        }
    }

    LogisticRegression lr(0);

    lr.Train(X, y);
    lr.Print();

    return 0;
}