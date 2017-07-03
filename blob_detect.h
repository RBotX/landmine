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
	float shield_height_ratio; //�߶������εı���
	int blur_size;             //ͼ��ƽ���뾶
	int min_area;              //��С���
	int max_area;			   //������
	int region_grow_thr;       //������������ֵ
	int centre_gray_thr;       //���ĵ�����
}blob_detect_params_t;


//���߼��
int blob_detect_process( Mat img, blob_detect_params_t* params, vector<blob_info_t>* pret );

//#ifdef __cplusplus
//}
//#endif 

#endif // blob_detect.h
