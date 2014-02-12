#include "LogisticRegression.h"
#include "../../../LOG.h"
#include <numeric>
#include <cassert>
#include <iostream>

using std::inner_product;
using std::cout;
using std::endl;

double sigmoid(double x)
{
    double e = 2.718281828;

    return 1.0 / (1.0 + pow(e, -x));
}

double dist(vector<double> v1, vector<double> v2)
{
    assert(v1.size() == v2.size());

    double sum = 0;

    for (int i = 0; i < v1.size(); i++)
    {
        sum += pow(v1[i] - v2[i], 2);
    }
    return sqrt(sum);
}

void LogisticRegression::Train(vector<vector<double>> X, vector<bool> y)
{
    assert(X.size() == y.size());

    theta.assign(X[0].size() + 1, 0.0); // add theta0 to head
    vector<double> old_theta(theta);

    int i;
    for (i = 0; i < max_iters; i++)
    {
        theta[0] -= alpha * (Predict(X[i]) - y[i]); // update theta0 at first position
        for (int j = 1; j < theta.size(); j++)
            theta[j] -= alpha * (Predict(X[i]) - y[i]) * X[i][j - 1];

        if (dist(theta, old_theta) < epsilon)
            break;

        old_theta = theta;
    }

    if (i == max_iters)
        LOG_DEBUG("i = " << i << '/' << max_iters);
    else
        LOG_DEBUG("i = " << i << '/' << max_iters << ", dist(" << dist(theta, old_theta) << ") < epsilon(" << epsilon << ")");
}

double LogisticRegression::Predict(vector<double> x)
{
    double z = inner_product(theta.begin() + 1, theta.end(), x.begin(), theta[0]);
    return sigmoid(z);
}

void LogisticRegression::Print()
{
    cout << "theta: ";
    for (int i = 0; i < theta.size(); i++)
        cout << theta[i] << ", ";
    cout << endl;
}