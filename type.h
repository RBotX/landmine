/**************************************************************
*
* @file   type.h
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

#ifndef __TYPE_H__
#define __TYPE_H__

#ifdef __cplusplus
extern "C"{
#endif 


typedef struct rect_tag
{
	int left;
	int right;
	int top;
	int bottom;
}rect_t;


typedef struct  blob_info_tag
{
	int status;             //状态
	double area;           //面积
	double angle;          //角度
	double major_axis;	   //主轴
	double minor_axis;     //次轴
	double hullArea;       //凸包的面积
	double perimeter;      //周长
	rect_t bd_rect;

}blob_info_t;


#ifndef MIN
#define MIN(a,b) (a)<(b)?(a):(b)
#endif 

#ifndef MAX
#define MAX(a,b) (a)>(b)?(a):(b)
#endif 

#ifndef ABS
#define ABS(a) (a)<0?-(a):(a)
#endif 



//#define _DEBUG_MODE  //调试模式开关
//#define _SIMPLE_COLLOECT_MODE  //样本采集模式开关


#ifdef __cplusplus
}
#endif 

#endif // type.h
