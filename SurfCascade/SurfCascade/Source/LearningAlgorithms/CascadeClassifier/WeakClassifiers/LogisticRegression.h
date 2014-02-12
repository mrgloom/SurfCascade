#ifndef LOGISTICREGRESSION_H
#define LOGISTICREGRESSION_H

#include "WeakClassifier.h"
#include <vector>

using std::vector;

class LogisticRegression : public WeakClassifier
{
    vector<double> theta;
    int max_iters = 100000;
    double alpha = 0.00001;
    double epsilon = 0.001;

public:
    LogisticRegression(int patch_index): WeakClassifier(patch_index) {};
    void Train(vector<vector<double>> X, vector<bool> y);
    double Predict(vector<double> X);
    void Print();
};

#endif