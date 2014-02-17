#ifndef STAGECLASSIFIER_H
#define STAGECLASSIFIER_H

#include "../WeakClassifiers/WeakClassifier.h"
#include <vector>
#include <memory>

using std::vector;
using std::shared_ptr;

class StageClassifier
{
    double search_step = 0.01;
    double auc_step = 0.05;
    double TPR_min;

protected:
    int n_total;
    int n_pos;
    int n_neg;

public:
    double FPR;
    double TPR;
    double theta;

    StageClassifier(double TPR_min_perstage): TPR_min(TPR_min_perstage) {}
    virtual void Train(vector<vector<vector<double>>> X, vector<bool> y) = 0;
    virtual double Predict(vector<vector<double>> x) = 0;
    double Evaluate(vector<vector<vector<double>>> X, vector<bool> y);
    void SearchTheta(vector<vector<vector<double>>> X, vector<bool> y);
};

#endif