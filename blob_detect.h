/**************************************************************
*
* @file   blob_detect.h
* @brief  This is a brief description.
* detail
*
* @author Justin
* @email  justin@hikdata.com
* @date   2017/04/06
* @version <version number>
*
* Copyright(c) HikData.2014-2017 .All rights reserved.
*
*-------------------------------------------------------------*
* change history:                                             *
* <date>      | <author>        | <description>               *
*-------------------------------------------------------------*
* 2017/04/06  | Justin     | Create File                 *
*-------------------------------------------------------------*
*
*/

#ifndef __BLOB_DETECT_H__
#define __BLOB_DETECT_H__

//#ifdef __cplusplus
//extern "C"{
//#endif


#include <vector>
#include "opencv2/core/core.hpp"
#include "type.h"

using namespace std;
using namespace cv;

typedef struct  blob_detect_params_tag
{
	float shield_height_ratio; //高度上屏蔽的比例
	int blur_size;             //图像平滑半径
	int min_area;              //最小面积
	int max_area;			   //最大面积
	int region_grow_thr;       //区域生长的阈值
	int centre_gray_thr;       //中心点亮度
}blob_detect_params_t;


//亮斑检测
int blob_detect_process( Mat img, blob_detect_params_t* params, vector<blob_info_t>* pret );

//#ifdef __cplusplus
//}
//#endif 

#endif // blob_detect.h
