#ifndef __TRACK_H__
#define __TRACK_H__

#include "opencv2/imgproc/imgproc.hpp"

int tracker_create( void** tracker );
int tracker_process(void* handle, cv::Mat cur_frame);

#endif 