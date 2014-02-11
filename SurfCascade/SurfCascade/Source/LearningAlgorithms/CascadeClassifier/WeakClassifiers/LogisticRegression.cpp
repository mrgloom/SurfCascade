#include "LogisticRegression.h"
#include <numeric>
#include <cassert>

using std::inner_product;

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

    theta.assign(X[0].size() + 1, 0.0); // append theta0 to tail
    vector<double> new_theta(theta);

    for (int k = 0; k < max_iters; k++)
    {
        for (int i = 0; i < X.size(); i++)
        {
            for (int j = 0; j < theta.size(); j++)
            {
                if (j == 0) // update theta0 at first position
                    new_theta[j] -= alpha * (Predict(X[i]) - y[i]);
                else
                    new_theta[j] -= alpha * (Predict(X[i]) - y[i]) * X[i][j - 1];
            }
        }

        if (dist(theta, new_theta) < epsilon)
        {
            new_theta.swap(theta);
            break;
        }

        theta = new_theta;
    }
}

double LogisticRegression::Predict(vector<double> x)
{
    double z = inner_product(theta.begin() + 1, theta.end(), x.begin(), theta[0]);
    return sigmoid(z);
}