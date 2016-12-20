#include <math.h>
#include <sys/time.h>
#include <stdint.h>
#include <string.h>

//TODO; We might want to change the implementation of the timer to use
//      state machine here (i.e. Start/Pause/Stop/Reset are states).

class Timer {

public:
    typedef struct {
        double   raw;
        uint32_t minute;
        uint32_t sec;
        uint32_t msec;
    } Duration;

    typedef enum {
        sec, msec
    } Format;
    
    Timer();
    ~Timer();

    void Start();
    void Pause();
    void Stop();
    void Reset();
    Duration GetDuration();
    Duration GetDuration(Format format);

private:
    timeval  mStartTimestamp;
    timeval  mEndTimestamp;
    Duration mDuration;
    bool     mRunning;
};

Timer::Timer() {
    mRunning = false;
}

Timer::~Timer() {
}

void Timer::Start() {
    if (mRunning)
        return;

    gettimeofday(&mStartTimestamp, NULL);
    mRunning = true;
}

void Timer::Pause() {
    if (!mRunning)
        return;

    gettimeofday(&mEndTimestamp, NULL);
    mRunning = false;
}

void Timer::Stop() {
    if (!mRunning)
        return;

    gettimeofday(&mEndTimestamp, NULL);
    mRunning = false;
}

void Timer::Reset() {
    memset(&mStartTimestamp, 0, sizeof(timeval));
    memset(&mEndTimestamp, 0, sizeof(timeval));
    mRunning = false;
}

Timer::Duration Timer::GetDuration() {
    return GetDuration(sec);
}

Timer::Duration Timer::GetDuration(Timer::Format format) {

    double rawStart = ((double) mStartTimestamp.tv_sec * 1000000.0) +
                        (double) mStartTimestamp.tv_usec;

    double rawEnd = ((double) mEndTimestamp.tv_sec * 1000000.0) +
                        (double) mEndTimestamp.tv_usec;

    double rawDuration = rawEnd - rawStart;
    mDuration.raw = rawDuration;

    double tempMin, tempSec, tempMsec;
    switch (format) {
    case sec:
        mDuration.minute = 0.0;
        tempSec = rawDuration / 1000000.0;
        tempMsec = tempSec - floor(tempSec);
        mDuration.sec = tempSec - tempMsec;
        mDuration.msec = tempMsec;
        break;
    case msec:
        mDuration.minute = 0.0;
        mDuration.sec = 0.0;
        mDuration.msec = rawDuration / 1000.0;
        break;
    }

    return mDuration;
}



