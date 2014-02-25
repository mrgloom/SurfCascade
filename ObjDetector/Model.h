#ifndef MODEL_H
#define MODEL_H

#include <string>

using std::string;

class CascadeClassifier;

class Model
{
public:
    string model_cfg;

    Model(string model_cfg) : model_cfg(model_cfg) {};
    ~Model();
    int Save(CascadeClassifier& cascade_classifier);
    int Load(CascadeClassifier& cascade_classifier);
    //void Load(CascadeClassifier& cascade_classifier);
};

#endif