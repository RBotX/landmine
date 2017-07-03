/**************************************************************
*
* @file   uav_detect.h
* @brief  This is a brief description.
* detail
*
* @author Justin
* @email  justin@hikdata.com
* @date   2017/06/06
* @version <version number>
*
* Copyright(c) HikData.2014-2017 .All rights reserved.
*
*-------------------------------------------------------------*
* change history:                                             *
* <date>      | <author>        | <description>               *
*-------------------------------------------------------------*
* 2017/06/06  | Justin     | Create File                 *
*-------------------------------------------------------------*
*
*/

#ifndef __UAV_DETECT_H__
#define __UAV_DETECT_H__


#include "opencv2/core/core.hpp"
#include <vector>

typedef struct uav_rect_tag
{
	int left;
	int top;
	int right;
	int bottom;
	int score;  // 0-10000
}uav_rect_t;

typedef std::vector<uav_rect_t>	vec_rect_t;

#ifdef __cplusplus
extern "C"{
#endif 

	int uav_detect_process(cv::Mat img, vec_rect_t* pret_vec );

#ifdef __cplusplus
}
#endif 

#endif // uav_detect.h