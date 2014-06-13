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
    param->C = 0.1;
    param->nr_weight = 0;

    set_print_string_function(&print_null);

    w = new float[33];
}

LogisticRegression::~LogisticRegression()
{
    delete[] w;
}

void LogisticRegression::Train(problem* prob)
{
    if (check_parameter(prob, param) == NULL) {
        model_ = train(prob, param);
        for (int i = 0; i < 33; i++)
            w[i] = (float)model_->w[i];
    }
    else
        LOG_ERROR("liblinear check_parameter() error.");
}

float LogisticRegression::Predict(vector<float>& x)
{
    double prob;

    // set each field to 0
    __m128 _s = _mm_set_ps1(0);

    for (int i = 0; i < x.size(); i += 4) {
        __m128 _t = _mm_mul_ps(_mm_loadu_ps(&w[i]), _mm_loadu_ps(&x.data()[i]));

        // add to four fields of s
        _s = _mm_add_ps(_t, _s);
    }
    // add up four fields of s
    _s = _mm_hadd_ps(_s, _s);
    _s = _mm_hadd_ps(_s, _s);
    prob = _s.m128_f32[0];

    prob += w[x.size()] * model_->bias;
    prob = 1 / (1 + exp(-prob));

    return (float)prob;
}
