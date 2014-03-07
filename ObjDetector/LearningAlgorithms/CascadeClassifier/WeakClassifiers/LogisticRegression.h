#ifndef LOGISTICREGRESSION_H
#define LOGISTICREGRESSION_H

#include "LearningAlgorithms/CascadeClassifier/WeakClassifiers/WeakClassifier.h"
#include <vector>

using std::vector;

class LogisticRegression : public WeakClassifier
{
    vector<double> theta;
    int max_iters = 1000;
    double alpha = 0.001;
    double epsilon = 0.0001;
    double lambda = 1;

public:
    LogisticRegression(int patch_index): WeakClassifier(patch_index) {};
    void Train(vector<vector<double>>& X, vector<bool>& y);
    double Predict(vector<double>& X);
    void Print();
    friend class Model;
};

#endif