#ifndef GENTLEADABOOST_H
#define GENTLEADABOOST_H

#include "StageClassifier.h"
#include <vector>

using std::vector;

class GentleAdaboost : public StageClassifier
{
    double total_AUC_score;
    int sample_num = 30 * 32;
    int max_iters = 100;
    vector<shared_ptr<WeakClassifier>> weak_classifiers;

public:
    GentleAdaboost(double TPR_min_perstage): StageClassifier(TPR_min_perstage) {}
    void Train(vector<vector<vector<double>>> X, vector<bool> y);
    double Predict(vector<vector<double>> x);
};

#endif