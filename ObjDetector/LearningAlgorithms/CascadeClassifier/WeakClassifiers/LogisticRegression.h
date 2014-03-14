#ifndef LOGISTICREGRESSION_H
#define LOGISTICREGRESSION_H

#include <vector>

using std::vector;

struct parameter;
struct problem;
struct model;
struct feature_node;

class LogisticRegression
{
    vector<double> theta;

    // for liblinear
    parameter* param;
    model* model_;

public:
    int patch_index;

    LogisticRegression(int patch_index);
    void Train(problem* prob);
    double Predict(vector<double>& x);
    friend class Model;
};

#endif