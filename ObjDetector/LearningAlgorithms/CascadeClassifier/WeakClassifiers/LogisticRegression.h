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
    // for liblinear
    parameter* param;
    model* model_;

public:
    int patch_index;

    LogisticRegression(int patch_index);
    void Train(problem* prob);
    float Predict(vector<float>& x);
    friend class Model;
};

#endif