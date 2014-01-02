#include "SurfCascade.h"
#include <windows.h>
#include <iostream>
#include <vector>
#include <stdio.h>

using namespace std;

int main(int argc, char *argv[])
{
    WIN32_FIND_DATA ffd;
    HANDLE hFind = INVALID_HANDLE_VALUE;
    vector<string> pos_files;

    hFind = FindFirstFile("D:\\test\\*", &ffd);
    if (hFind == INVALID_HANDLE_VALUE) return -1;
    do
    {
        pos_files.push_back(string(ffd.cFileName));
    } while (FindNextFile(hFind, &ffd) != 0);
    FindClose(hFind);

    return 0;
}