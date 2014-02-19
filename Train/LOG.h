#ifndef LOG_H
#define LOG_H

#include <iostream>

using std::cout;
using std::endl;
using std::flush;

enum LEVELS { ERROR_LEVEL, WARNING_LEVEL, INFO_LEVEL, DEBUG_LEVEL };

#define SETLEVEL INFO_LEVEL

// append newline
#define LOG_DEBUG(MSG)   do { if (SETLEVEL >= DEBUG_LEVEL)   cout << /*__FILE__ << ':' << __FUNCTION__ << ':' << __LINE__ << ":DEBUG: "   << */ MSG << endl; } while(0)
#define LOG_INFO(MSG)    do { if (SETLEVEL >= INFO_LEVEL)    cout << /*__FILE__ << ':' << __FUNCTION__ << ':' << __LINE__ << ":INFO: "    << */ MSG << endl; } while(0)
#define LOG_WARNING(MSG) do { if (SETLEVEL >= WARNING_LEVEL) cout << /*__FILE__ << ':' << __FUNCTION__ << ':' << __LINE__ << ":WARNING: " << */ MSG << endl; } while(0)
#define LOG_ERROR(MSG)   do { if (SETLEVEL >= ERROR_LEVEL)   cout << /*__FILE__ << ':' << __FUNCTION__ << ':' << __LINE__ << ":ERROR: "   << */ MSG << endl; } while(0)

// no newline
#define LOG_DEBUG_NN(MSG)   do { if (SETLEVEL >= DEBUG_LEVEL)   cout << /*__FILE__ << ':' << __FUNCTION__ << ':' << __LINE__ << ":DEBUG: "   << */ MSG; } while(0)
#define LOG_INFO_NN(MSG)    do { if (SETLEVEL >= INFO_LEVEL)    cout << /*__FILE__ << ':' << __FUNCTION__ << ':' << __LINE__ << ":INFO: "    << */ MSG; } while(0)
#define LOG_WARNING_NN(MSG) do { if (SETLEVEL >= WARNING_LEVEL) cout << /*__FILE__ << ':' << __FUNCTION__ << ':' << __LINE__ << ":WARNING: " << */ MSG; } while(0)
#define LOG_ERROR_NN(MSG)   do { if (SETLEVEL >= ERROR_LEVEL)   cout << /*__FILE__ << ':' << __FUNCTION__ << ':' << __LINE__ << ":ERROR: "   << */ MSG; } while(0)

#endif