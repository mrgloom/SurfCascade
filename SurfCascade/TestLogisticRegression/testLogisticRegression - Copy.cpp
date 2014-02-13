#include "../SurfCascade/Source/LearningAlgorithms/CascadeClassifier/WeakClassifiers/LogisticRegression.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <string>

using namespace std;

int main()
{
    ifstream f("D:/weekly.csv");
    //ifstream f("D:/default1.csv");
    //ifstream f("D:/smarket.csv");
    string line;
    vector<vector<double>> X = { { 1, 1 }, { 2, 1 }, { 1, 6 }, { 3, 4 }, { 5, 2 }, { 7, 9 }, { 8, 3 }, { 1.5, 6 }, { 10, 11 } };
    vector<double> y = { 3, 4, 13, 11, 9, 25, 14, 13.5, 32 };
    double a;

    //while (getline(f, line))
    //{
    //    X.push_back(vector<double>());
    //    stringstream ss(line);
    //    while (ss >> a)
    //    {
    //        if (ss.peek() == ',')
    //        {
    //            X.back().push_back(a);
    //            ss.ignore();
    //        }
    //        else
    //            y.push_back((bool)a);
    //    }
    //}

    LogisticRegression lr(0);

    lr.Train(X, y);
    lr.Print();

    return 0;
}