#ifndef WEAKCLASSIFIER_H
#define WEAKCLASSIFIER_H

#include <vector>

using std::vector;

class WeakClassifier
{
public:
    int patch_index;
    WeakClassifier(int patch_index) :patch_index(patch_index) {};
    virtual void Train(vector<vector<double>> X, vector<bool> y) = 0;
    virtual double Predict(vector<double> x) = 0;
};

#endif