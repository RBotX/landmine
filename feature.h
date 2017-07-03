/**************************************************************
*
* @file   feature.h
* @brief  This is a brief description.
* detail
*
* @author Justin
* @email  justin@hikdata.com
* @date   2017/04/02
* @version <version number>
*
* Copyright(c) HikData.2014-2017 .All rights reserved.
*
*-------------------------------------------------------------*
* change history:                                             *
* <date>      | <author>        | <description>               *
*-------------------------------------------------------------*
* 2017/04/02  | Justin     | Create File                 *
*-------------------------------------------------------------*
*
*/

#ifndef __FEATURE_H__
#define __FEATURE_H__


#include "type.h"

#ifdef __cplusplus
extern "C"{
#endif 

	int calc_lbp_map(unsigned char* pimg, unsigned char* plbp_map, int img_h, int img_w);

	//��ȡͼ��roi�����ڵ�lbpֱ��ͼ
	// proi_rectΪȫ������
	int calc_lbp_feat_roi(unsigned char* plbp_map, int img_w, float *phist, int bin_nr, rect_t* proi_rect);


	//�����ݶ�ֵ�ͽǶ�
	int calc_hog_gradient_img(unsigned char* pimg, unsigned char* pgrad_mag, unsigned char* pgrad_ang, int img_h, int img_w);

	int calc_hog_feat_roi(unsigned char* pgrad_mag, unsigned char* pgrad_ang, int img_w, float *phist, int bin_nr, rect_t* proi_rect);


#ifdef __cplusplus
}
#endif 

#endif // feature.h
