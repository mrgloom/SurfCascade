#include "LogisticRegression.h"
#include "../../../LOG.h"
#include <numeric>
#include <cassert>
#include <iostream>
#include <cmath>

using std::inner_product;
using std::cout;
using std::endl;

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

    double diff;
    int k, i;
    for (k = 0; k < max_iters; k++)
    {
        for (i = 0; i < X.size(); i++)
        {
            diff = y[i] - Predict(X[i]);

            theta[0] += alpha * diff; // update theta0 at first position
            for (int j = 1; j < theta.size(); j++)
                theta[j] += alpha * diff * X[i][j - 1];

            //if (diff > 0.0000000000000001)
            if (dist(theta, old_theta) < epsilon)
            {
                LOG_DEBUG("\t" << "k = " << k << '/' << max_iters << ", i = " << i << '/' << X.size() << ", dist(" << dist(theta, old_theta) << ") < epsilon(" << epsilon << ")");
                return;
            }

            old_theta = theta;
        }
    }

    LOG_DEBUG("\tk = " << k << '/' << max_iters << ", i = " << i << '/' << X.size());
}

double LogisticRegression::Predict(vector<double> x)
{
    double z = inner_product(theta.begin() + 1, theta.end(), x.begin(), theta[0]);

    return 1.0 / (1.0 + exp(-z));
}

void LogisticRegression::Print()
{
    cout << "theta: " << endl;
    for (int i = 0; i < theta.size(); i++)
        cout << theta[i] << endl;
    cout << endl;
}