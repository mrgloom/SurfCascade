#include "LearningAlgorithms/CascadeClassifier/WeakClassifiers/LogisticRegression.h"
#include "LOG.h"
#include "linear.h"
#include <numeric>
#include <cassert>
#include <iostream>
#include <cmath>

using std::inner_product;
using std::cout;
using std::endl;

void print_null(const char *s) {}

LogisticRegression::LogisticRegression(int patch_index)
{
    this->patch_index = patch_index;

    param = new parameter();
    param->solver_type = 0;
    param->eps = 0.01;
    param->C = 1;
    param->nr_weight = 0;

    set_print_string_function(&print_null);
}

void LogisticRegression::Train(problem* prob)
{
    if (check_parameter(prob, param) == NULL)
        model_ = train(prob, param);
    else
        LOG_ERROR("liblinear check_parameter() error.");
}

double LogisticRegression::Predict(vector<double>& x)
{
    double prob_estimates[2];

    feature_node* fn = new feature_node[x.size() + 2];

    int i;
    for (i = 0; i < x.size(); i++)
    {
        fn[i].index = i;
        fn[i].value = x[i];
    }
    fn[i].index = i;
    fn[i].value = 1;
    fn[++i].index = -1;

    predict_probability(model_, fn, prob_estimates);

    delete[] fn;

    return prob_estimates[0];
}