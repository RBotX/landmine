// landmine.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "cv.h"
#include "highgui.h"
#include "opencv2/features2d/features2d.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "feature.h"
#include "type.h"
#include "get_file_features.h"
#include "svm.h"
#include "blob_detect.h"
#include "uav_detect.h"
#include "track.h"

#include "opencv2/video/tracking.hpp"

#define min(a,b) (a)<(b)?(a):(b)
#define max(a,b) (a)>(b)?(a):(b)
#ifndef ABS
#define ABS(a) (a)<0?-(a):(a)
#endif 


#define TEST_COLOR_INVERSE  1             //颜色翻转测试
#define SHIELD_HEIGHT_RATIO (0)	          //高度屏蔽的比例
#define DRAW_FAULT_POINT    (1)		      //绘制虚假的点
#define TEST_VIDEO          (1)		      //测试视频效果,否则为图片测试
//#define PIC_SAVE_PATH           "d://test_img/"	  //结果保存路径



static void copy_image_from_cv(unsigned char* pdata,IplImage*pimg )
{
	int i,j,k;
	k = 0;
	for (i=0;i<pimg->height;i++)
	{
		for (j=0;j<pimg->width;j++)
		{
			CvScalar val = cvGet2D(pimg,i,j);
			pdata[k] = val.val[0];
			k++;
		}
	}
}

static void copy_image_to_cv(unsigned char* pdata,IplImage*pimg )
{
	int i,j,k;
	k = 0;
	for (i=0;i<pimg->height;i++)
	{
		for (j=0;j<pimg->width;j++)
		{
			cvSet2D(pimg,i,j,cvScalarAll(pdata[k]));
			k++;
		}
	}
}

static void overlap_cv_image( unsigned char* pdata,IplImage*pimg )
{
	int i,j,k;
	k = 0;
	for (i=0;i<pimg->height;i++)
	{
		for (j=0;j<pimg->width;j++)
		{
			if(pdata[k]>128)
			{
				CvScalar val = cvGet2D(pimg,i,j);
				cvSet2D(pimg,i,j,cvScalar(0.5*val.val[0],0.5*val.val[1],0.5*val.val[0] + 128));
			}
			k++;
		}
	}
}


//
static int bernsen_binary(unsigned char*pimg, unsigned char* pbin, int img_w, int img_h)
{
	int win_r = 7;
	int win_diff_th = 30;
	int min_thr = 0;
	int i,j;
	int m,n;

	memset( pbin,0,sizeof(*pbin)*img_h*img_w );

	for (i=win_r;i<img_h-win_r;i++)
	{
		for (j=win_r;j<img_w-win_r;j++)
		{
			int win_min = pimg[i*img_w+j];
			int win_max = win_min;
			int hist[256];
			int area_thr = 0.2*(2*win_r+1)*(2*win_r+1);
			int acc_min,acc_max;

			memset( hist,0,sizeof(hist));
			for (m=-win_r;m<=win_r;m++)
			{
				for (n=-win_r;n<=win_r;n++)
				{
					int v = pimg[(i+m)*img_w+(j+n)];
					hist[v]++;
					//win_min = min( win_min,v);
					//win_max = max( win_min,v);
				}
			}

			acc_min = acc_max = 0;
			for (m=0;m<256;m++)
			{
				acc_min += hist[m];
				if( acc_min>area_thr )
				{
					win_min = m;
					break;
				}
			}

			for (m=255;m>=0;m--)
			{
				acc_max += hist[m];
				if( acc_max>area_thr )
				{
					win_max = m;
					break;
				}
			}

			if( ( win_max-win_min) > win_diff_th && pimg[i*img_w+j]>min_thr )
			{
				pbin[i*img_w+j] = 255;
			}

		}
	}
	return 0;
}

//1.数据平滑处理
//2.增加一个突变点检测

//主处理函数
static int process(unsigned char* pdata, unsigned char* pout, int img_w, int img_h)
{
	int i,j;
	//memcpy( pout,pdata,img_h*img_w );
	bernsen_binary( pdata,pout,img_w,img_h );
	return 0;
}

static int fun(char* filename)
{
	IplImage *pimg = cvLoadImage(filename,CV_LOAD_IMAGE_COLOR);
	IplImage *poutimg = NULL;
	unsigned char* pimg_data = NULL;
	unsigned char* pout = NULL;

	if(pimg==NULL) goto FUNC_EXIT;
	pimg_data = (unsigned char*)malloc( pimg->height*pimg->width );
	pout = (unsigned char*)malloc( pimg->height*pimg->width );
	//cvCreateImage(cvGetSize(pimg),8,pimg->nChannels);
	poutimg = cvCloneImage( pimg );
	

	//cvAdaptiveThreshold();
	copy_image_from_cv( pimg_data,pimg );
	process( pimg_data,pout,pimg->width,pimg->height );

	//copy_image_to_cv( pout,poutimg );
	overlap_cv_image(pout,poutimg);

	cvNamedWindow("IMG",0);
	cvShowImage("IMG",pimg);

	cvNamedWindow("OUT",0);
	cvShowImage("OUT",poutimg);

	cvWaitKey(0);
	cvReleaseImage(&pimg);
	cvReleaseImage(&poutimg);
	free(pimg_data);
	free(pout);

FUNC_EXIT:

	return 0;
}

#if 0
int _tmain(int argc, _TCHAR* argv[])
{
	char filename[260] = {"E:\\HIKDATA\\data\\2016年冬场景记录表\\草地\\马上试验场红外区域.jpg"};

	do 
	{
		printf("file:");
		scanf("%s",filename);
		fun(filename);
	} while (1);
}
#else

#include "opencv2/core/core.hpp"  
#include "highgui.h"  
#include "opencv2/imgproc/imgproc.hpp"  
#include "opencv2/features2d/features2d.hpp"  
#include "opencv2/nonfree/nonfree.hpp" 
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"

using namespace cv;  
//using namespace std;  



//计算积分图
static void calc_integral_img(unsigned char* pimg,int* pintegral,int img_w,int img_h)
{
	int i,j;
	int* ptmp_int = pintegral;
	pintegral[0] = pimg[0];
	for (i=1;i<img_w;i++)
	{
		pintegral[i] = pimg[i] + pintegral[i-1];
	}


	pimg += img_w;
	ptmp_int += img_w;
	for (i=1;i<img_h;i++)
	{
		int sum = 0;
		for (j=0;j<img_w;j++)
		{
			sum  += pimg[j];
			ptmp_int[j] = pintegral[j] + sum;
		}

		ptmp_int += img_w;
		pintegral += img_w;
		pimg += img_w;
	}
}


//计算积分图的值
static int calc_block_sum(int* pintegral,int left,int right,int top,int bottom ,int img_w )
{
	int val = pintegral[top*img_w+left] + pintegral[bottom*img_w+right] - pintegral[top*img_w+right] - pintegral[bottom*img_w+left];
	return val/( (right-left+1)*(bottom-top+1) );
}


//计算候选点附近的对比特性
static void calc_hat_feat(unsigned char* pimg, vector<KeyPoint>* keypoints, int img_w, int img_h)
{
	int k;
	vector<KeyPoint>::iterator iter;
	int* pintegral = new int[img_h*img_w];
	float bd_plus = 2;
	
	calc_integral_img( pimg,pintegral,img_w,img_h );

	for (iter=keypoints->begin();iter!=keypoints->end();iter++)
	{
		int thr = 15;
		int x = iter->pt.x;
		int y = iter->pt.y;
		float size = max(5,iter->size);
		int left,top,right,bottom;
		int radius = max(7,min(15,size*bd_plus ));
		int val[8];

		//if (iter->response<1) continue;

		iter->response = 0;

		if( (x + (radius + size )) >= (img_w-3) ||
			(y + (radius + size )) >= (img_h-3) ||
			(x - (radius + size )) <= 3 ||
			(y - (radius + size )) <= 3 )
		{
			continue;
		}

		if (y < img_h*SHIELD_HEIGHT_RATIO)
		{
			continue;
		}

		//计算模块均值
		left = x - size;
		right = x + size;
		top = y - size;
		bottom = y + size;
		
		val[0] = calc_block_sum( pintegral,left-radius,left,        top-radius,top,img_w );
		val[1] = calc_block_sum( pintegral,left,       right,       top-radius,top,img_w );
		val[2] = calc_block_sum( pintegral,right,      right+radius,top-radius,top,img_w );

		val[3] = calc_block_sum( pintegral,left-radius,left,        top,       bottom,img_w );
		int v11 = calc_block_sum( pintegral,left,       right,       top,       bottom,img_w );
		val[4] = calc_block_sum( pintegral,right,      right+radius,top,       bottom,img_w );

		val[5] = calc_block_sum( pintegral,left-radius,left,        bottom,    bottom+radius,img_w );
		val[6] = calc_block_sum( pintegral,left,       right,       bottom,    bottom+radius,img_w );
		val[7] = calc_block_sum( pintegral,right,      right+radius,bottom,    bottom+radius,img_w );

		//比较阈值
		thr = max( (v11*0.15),20 );
		int thr_low = max( (v11*0.1),10 );

		int big_nr =  0;//
		int big_nr_low =  0;//

		for (k=0;k<8;k++)
		{
			big_nr += v11>(val[k]+thr);
			big_nr_low += v11>(val[k]+thr_low);
		}
		
		if(big_nr_low>=8 && big_nr>=1)
		{
			iter->response = 1;
		}
		

	}

	delete [] pintegral;
}

static int blob_detect(char* filename)  
{  
	int i;
	Mat img_rgb ;
	Mat img_gray;
	Mat lp_img;
	Mat lp_img_u8;
	Mat img_bak;
	int kerner_size = 3;
	if(filename)
	{
		img_rgb = imread(filename);
	}
	else 
	{
		img_rgb = imread("E:\\HIKDATA\\data\\2016年冬场景记录表\\草地\\马上试验场红外区域.jpg");
	}


	//Mat img = imread("c:\\1.png"); 
	//if(img.channels()>=3)
	{
		cv::cvtColor(img_rgb, img_gray, COLOR_RGB2GRAY);
	}

	img_bak = img_gray.clone();
	cv::GaussianBlur(img_gray, img_gray, cv::Size(kerner_size, kerner_size), 2);
	cv::Laplacian(img_gray, lp_img, CV_16S, 3);

	lp_img_u8 = img_gray.clone();
	short* lps16_ptr = (short*)lp_img.data;
	unsigned char* lpu8_ptr = (unsigned char*)lp_img_u8.data;
	for (i=0;i<lp_img_u8.rows*lp_img_u8.cols;i++)
	{
		lpu8_ptr[i] = lps16_ptr[i]<0?min(255,-2*lps16_ptr[i]):0;
	}

	//convertScaleAbs( lp_img, lp_img_u8 );

#if TEST_COLOR_INVERSE
	img_gray = 255 - img_gray;  //测试反色之后的效果
#endif 

	Mat output_img;  
	vector<KeyPoint> keypoints;    
	SimpleBlobDetector::Params params; 

	params.minThreshold = 20;
	params.maxThreshold = 220;
	params.filterByArea = true;  
	params.minArea = 4*4;  
	params.maxArea = 30*30; 

	params.filterByColor = 1;
	params.blobColor = TEST_COLOR_INVERSE?0:255;

	SimpleBlobDetector blobDetector( params );    
	blobDetector.create("SimpleBlob"); 


	blobDetector.detect(img_gray, keypoints);
	vector<KeyPoint>::iterator iter;
	output_img = 255 - img_gray;
	calc_hat_feat( output_img.data,&keypoints,output_img.cols,output_img.rows);

	for ( iter=keypoints.begin();iter!=keypoints.end();iter++ )
	{
        #if DRAW_FAULT_POINT
        #else 
		if (iter->response)
        #endif
		{
			circle(img_rgb, iter->pt, iter->size, iter->response ? Scalar(0, 0, 255) : Scalar(0, 0, 0), 1, 8, 0);
		}
	}

	
	namedWindow("result", 1);   
	imshow("result", img_rgb);

	namedWindow("LOG", 1);   
	imshow("LOG", lp_img_u8);  

#if 0
	{
		char savename[260];
		char ext_file[] = "_detect.jpg";
		strcpy( savename,filename );
		strcat( savename,ext_file);
		imwrite( string(savename),output_img);
	}
#endif 
	
	waitKey(0);   

	return 0;  
}

//区域生长
static  Mat seed_region_group(unsigned char* pimg, KeyPoint* key_pt, rect_t* rect, int img_w, int img_h, int th, int *region_area, int *region_avg_val)
{
	int area = 0;
	int avg_val = 0;
	Point2i pt(key_pt->pt.x, key_pt->pt.y);
	Point2i ptGrowing;						//待生长点位置
	int nGrowLable = 0;								//标记是否生长过
	int nSrcValue = 0;								//生长起点灰度值
	int nCurValue = 0;								//当前生长点灰度值
	Mat matDst = Mat::zeros(cv::Size(img_w,img_h), CV_8UC1);	//创建一个空白区域，填充为黑色

	//生长方向顺序数据
	int DIR[8][2] = { { -1, -1 }, { 0, -1 }, { 1, -1 }, { 1, 0 }, { 1, 1 }, { 0, 1 }, { -1, 1 }, { -1, 0 } };
	Vector<Point2i> vcGrowPt;						//生长点栈
	vcGrowPt.push_back(pt);							//将生长点压入栈中
	matDst.at<uchar>(pt.y, pt.x) = 255;				//标记生长点
	nSrcValue = pimg[pt.y*img_w+pt.x];			//记录生长点的灰度值

	while (!vcGrowPt.empty())						//生长栈不为空则生长
	{
		pt = vcGrowPt.back();						//取出一个生长点
		vcGrowPt.pop_back();

		//分别对八个方向上的点进行生长
		for (int i = 0; i < 9; ++i)
		{
			ptGrowing.x = pt.x + DIR[i][0];
			ptGrowing.y = pt.y + DIR[i][1];
			//检查是否是边缘点
			if (ptGrowing.x < rect->left || ptGrowing.y < rect->top || ptGrowing.x > rect->right || ptGrowing.y > rect->bottom )
				continue;

			nGrowLable = matDst.at<uchar>(ptGrowing.y, ptGrowing.x);		//当前待生长点的灰度值

			if (nGrowLable == 0)					//如果标记点还没有被生长
			{
				nCurValue = pimg[ptGrowing.y*img_w+ptGrowing.x];
				if (abs(nSrcValue - nCurValue) < th)					//在阈值范围内则生长
				{
					matDst.at<uchar>(ptGrowing.y, ptGrowing.x) = 255;		//标记为白色
					vcGrowPt.push_back(ptGrowing);					//将下一个生长点压入栈中

					area++;
					avg_val += nCurValue;
				}
			}
		}
	}

	*region_area = area;
	*region_avg_val = avg_val / area;

	return matDst.clone();
}

#if 0
typedef struct  blob_info_tag
{
	double area;           //面积
	double angle;
	double major_axis;	    //主轴
	double minor_axis;     //次轴
	double hullArea;       //凸包的面积
	double perimeter;      //周长
	rect_t bd_rect;

}blob_info_t;
#endif 


typedef struct  Center_tag
{
	Point2d location;
	double radius;
	double confidence;
	blob_info_t info;
}Center;



static void findBlobs(const cv::Mat &binaryImage, vector<Center> &centers)
{
	centers.clear();

	vector < vector<Point> > contours;
	Mat tmpBinaryImage = binaryImage.clone();
	findContours(tmpBinaryImage, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

	if (contours.size() > 1) return;

#ifdef DEBUG_BLOB_DETECTOR
	//  Mat keypointsImage;
	//  cvtColor( binaryImage, keypointsImage, CV_GRAY2RGB );
	//
	//  Mat contoursImage;
	//  cvtColor( binaryImage, contoursImage, CV_GRAY2RGB );
	//  drawContours( contoursImage, contours, -1, Scalar(0,255,0) );
	//  imshow("contours", contoursImage );
#endif

	for (size_t contourIdx = 0; contourIdx < contours.size(); contourIdx++)
	{
		Center center;
		center.confidence = 1;
		Moments moms = moments(Mat(contours[contourIdx]));

		if (1)//params.filterByArea)
		{
			double area = moms.m00;

			center.info.area = area;

			//if (area < params.minArea || area >= params.maxArea)
			//	continue;
		}

		if (1)//params.filterByCircularity)
		{
			double area = moms.m00;
			double perimeter = arcLength(Mat(contours[contourIdx]), true);
			double ratio = 4 * CV_PI * area / (perimeter * perimeter);

			center.info.perimeter = perimeter;

			//if (ratio < params.minCircularity || ratio >= params.maxCircularity)
			//	continue;
		}

		if (1)//params.filterByInertia)
		{
#if 0
			double denominator = sqrt(pow(2 * moms.mu11, 2) + pow(moms.mu20 - moms.mu02, 2));
			const double eps = 1e-2;
			double ratio;
			if (denominator > eps)
			{
				double cosmin = (moms.mu20 - moms.mu02) / denominator;
				double sinmin = 2 * moms.mu11 / denominator;
				double cosmax = -cosmin;
				double sinmax = -sinmin;

				double imin = 0.5 * (moms.mu20 + moms.mu02) - 0.5 * (moms.mu20 - moms.mu02) * cosmin - moms.mu11 * sinmin;
				double imax = 0.5 * (moms.mu20 + moms.mu02) - 0.5 * (moms.mu20 - moms.mu02) * cosmax - moms.mu11 * sinmax;
				ratio = imin / imax;

				center.info.minor_axis = imin;
				center.info.major_axis = imax;
			}
			else
			{
				ratio = 1;

				center.info.minor_axis = 0;
				center.info.major_axis = 0;
			}
#endif 

			RotatedRect fit = fitEllipse(contours[contourIdx]);
			center.info.minor_axis = fit.size.width;
			center.info.major_axis = fit.size.height;
			center.info.angle = fit.angle;
			//if (ratio < params.minInertiaRatio || ratio >= params.maxInertiaRatio)
			//	continue;

			center.confidence = 0;// ratio * ratio;
		}

		if (1 )//params.filterByConvexity)
		{
			vector < Point > hull;
			convexHull(Mat(contours[contourIdx]), hull);
			double area = contourArea(Mat(contours[contourIdx]));
			double hullArea = contourArea(Mat(hull));
			double ratio = area / hullArea;

			center.info.hullArea = hullArea;

			//if (ratio < params.minConvexity || ratio >= params.maxConvexity)
			//	continue;
		}

		center.location = Point2d(moms.m10 / moms.m00, moms.m01 / moms.m00);

		if (0 ) //params.filterByColor)
		{
			//if (binaryImage.at<uchar>(cvRound(center.location.y), cvRound(center.location.x)) != params.blobColor)
			//	continue;
		}

		//compute blob radius
		{
			vector<double> dists;
			rect_t rect_roi;
			Point2d pt = contours[contourIdx][0];
			rect_roi.left = rect_roi.right = pt.x;
			rect_roi.top = rect_roi.bottom = pt.y;
			for (size_t pointIdx = 0; pointIdx < contours[contourIdx].size(); pointIdx++)
			{
				pt = contours[contourIdx][pointIdx];
				rect_roi.left  = MIN(rect_roi.left,pt.x);
				rect_roi.right = MAX(rect_roi.right, pt.x);
				rect_roi.top   = MIN(rect_roi.top, pt.y);
				rect_roi.bottom= MAX(rect_roi.bottom, pt.y);
			}

			center.info.bd_rect = rect_roi;
			//rect_roi
		}

		centers.push_back(center);

#ifdef DEBUG_BLOB_DETECTOR
		//    circle( keypointsImage, center.location, 1, Scalar(0,0,255), 1 );
#endif
	}
#ifdef DEBUG_BLOB_DETECTOR
	//  imshow("bk", keypointsImage );
	//  waitKey();
#endif
}

//static 
static int counter = 0; 
//计算候选点附近的对比特性
static void region_group(Mat gray_img, vector<KeyPoint>* keypoints, int img_w, int img_h, vector<blob_info_t>* blob_info_vec)
{
	int i, j;
	vector<KeyPoint>::iterator iter;
	vector<KeyPoint> bak_pts;
	unsigned char* pimg = gray_img.data;

	Mat matDst = Mat::zeros(cv::Size(img_w, img_h), CV_8UC1);	//创建一个空白区域，填充为黑色
	int thr = 30;
	char filename[260];

	if ( blob_info_vec )
	{
		blob_info_vec->clear();
	}

	for (iter = keypoints->begin(); iter != keypoints->end(); iter++)
	{
		KeyPoint key_pt = *iter;
		rect_t rect_roi, rect_roi_sub;
		int k = 5;
		int area, avg_val;

		if( key_pt.response==0 ) continue;
		key_pt.response = 0;

		rect_roi.left  = MAX( 0, key_pt.pt.x - k * key_pt.size );
		rect_roi.right = MIN( img_w-1,key_pt.pt.x + k * key_pt.size );
		rect_roi.top   = MAX(0,key_pt.pt.y - k * key_pt.size);
		rect_roi.bottom= MIN( img_h-1,key_pt.pt.y + k * key_pt.size );

		Mat cur_mask = seed_region_group(pimg, &key_pt, &rect_roi, img_w, img_h, thr,&area,&avg_val);

		//得到局部小图,然后对小图进行findcounter操作
		Mat roi_mat = cur_mask( Rect(rect_roi.left, rect_roi.top, rect_roi.right - rect_roi.left, rect_roi.bottom - rect_roi.top) );
		Mat roi_img;

		//统计图像内部的方差和直方图,删除方差过大或者两个峰值距离较大的候选


		//对该部分图像进行分析
		vector<Center> centers;
		findBlobs(roi_mat, centers);

		if (centers.size() < 1)
		{
			key_pt.response = 0;
			continue;
		}

		//
		if (avg_val<128) continue;


		Center center = centers[0];

		int rect_w = center.info.bd_rect.right - center.info.bd_rect.left + 1;
		int rect_h = center.info.bd_rect.bottom - center.info.bd_rect.top + 1;
		int search_rect_w = rect_roi.right - rect_roi.left;
		int search_rect_h = rect_roi.bottom - rect_roi.top;

		center.info.area = area;

		//如果填充太大
		if ((rect_w*rect_h) > 0.5* (search_rect_w*search_rect_h) || 
			center.info.area > 25 * 25 ||
			rect_w > 0.5*search_rect_w ||
			rect_h > 0.5*search_rect_h )
		{
			key_pt.response = 0;
			//goto BREAK_LOOP;
			continue;
		}

		if ( rect_w < 10 && rect_h < 10 )
		{
			key_pt.response = 0;
			//goto BREAK_LOOP;
			continue;
		}

		//如果
		if ( rect_w >= 10 && rect_h >= 10 )
		{
			if (center.info.area < 0.7* (rect_w*rect_h))
			{
				key_pt.response = 0;
				//goto BREAK_LOOP;
				continue;
			}
		}
		

		if ( (center.info.major_axis / center.info.minor_axis) < 2)
		{
			key_pt.response = 1;
		}

		key_pt.size = MAX(rect_w, rect_h) / 2;


		//if (fabs(center.info.angle - 90) < 30)
		//{
		//	key_pt.response = 1;
		//}

		//blobDetector->findBlob()
BREAK_LOOP:
		bak_pts.push_back(key_pt);


		//抠图处理
		float ratio = 0.5;

		rect_roi_sub.left = MAX(0, rect_roi.left + center.info.bd_rect.left - rect_w*ratio);
		rect_roi_sub.right = MIN(img_w - 1, rect_roi.left + center.info.bd_rect.left + rect_w + rect_w*ratio);
		rect_roi_sub.top = MAX(0, rect_roi.top + center.info.bd_rect.top - rect_h*ratio);
		rect_roi_sub.bottom = MIN(img_h - 1, rect_roi.top + center.info.bd_rect.top + rect_h + rect_h*ratio);

		roi_mat = cur_mask(Rect(rect_roi_sub.left, rect_roi_sub.top, rect_roi_sub.right - rect_roi_sub.left + 1, rect_roi_sub.bottom - rect_roi_sub.top + 1));
		roi_img = gray_img(Rect(rect_roi_sub.left, rect_roi_sub.top, rect_roi_sub.right - rect_roi_sub.left + 1, rect_roi_sub.bottom - rect_roi_sub.top + 1));

		if (blob_info_vec && key_pt.response )
		{
			blob_info_t blob_info;
			blob_info = center.info;
			blob_info.bd_rect = rect_roi_sub;
			blob_info_vec->push_back(blob_info);
		}

		//counter++;
		//sprintf(filename, "d:\\imgs\\%s\\mask_%d.jpg", key_pt.response ? "pos" : "neg", counter);
		//imwrite(filename, roi_mat);

		//sprintf(filename, "d:\\imgs\\%s\\img_%d.jpg", key_pt.response ? "pos" : "neg", counter);
		//imwrite(filename, roi_img);

		matDst |= cur_mask;
	}


	keypoints->clear();
	for (iter = bak_pts.begin(); iter != bak_pts.end(); iter++)
	{
		KeyPoint key_pt = *iter;
		keypoints->push_back(key_pt);
	}


	IplImage* prgb_img = cvCreateImage(cvSize(img_w, img_h), 8, 3);
	float ratio = 0.7;
	for (i = 0; i < img_h;i++)
	{
		for (j = 0; j < img_w;j++)
		{
			cvSet2D(prgb_img, i, j, cvScalarAll(pimg[i*img_w+j]));
			if (matDst.data[i*img_w + j])
			{
				cvSet2D(prgb_img, i, j, cvScalar(255 * (1 - ratio) + ratio*pimg[i*img_w + j], ratio*pimg[i*img_w + j], ratio*pimg[i*img_w + j], 0));
			}
		}
	}

	cvNamedWindow("GROUP", 0);
	cvShowImage("GROUP", prgb_img);
	cvReleaseImage(&prgb_img);

}

static  int create_detector( SimpleBlobDetector** blobDetector )
{
	SimpleBlobDetector::Params params;

	//memset( &params, 0, sizeof(params));

	params.minThreshold = 20;
	params.maxThreshold = 200;
	params.thresholdStep = 5;
	params.filterByArea = true;
	params.minArea = 4 * 4;
	params.maxArea = 25 * 25;

	params.filterByColor = 1;
	params.blobColor = TEST_COLOR_INVERSE ? 0 : 255;

	params.filterByConvexity = 0;  //凸率
	params.filterByInertia = 0;	   //惯性比
	params.minInertiaRatio = 0.5;
	params.maxInertiaRatio = 2;
	params.filterByCircularity = 0;

	*blobDetector = new SimpleBlobDetector(params);
	(*blobDetector)->create("SimpleBlob");

	return 0;
}


static int proc_func(cv::Mat& img_rgb, SimpleBlobDetector* blobDetector, int fram_nr, vector<KeyPoint>& keypoints, vector<blob_info_t>& blob_info)
{
	Mat img_gray, img_bak;// lp_img, lp_img_u8;
	int kerner_size = 3;
	int i,j;
	short* lps16_ptr;
	unsigned char* lpu8_ptr;
	char save_filename[260];

	Mat output_img;
	//Mat bin_img;
	vector<KeyPoint>::iterator iter;

	cv::cvtColor(img_rgb, img_gray, COLOR_RGB2GRAY);
	img_bak = img_gray.clone();
	cv::GaussianBlur(img_gray, img_gray, cv::Size(kerner_size, kerner_size), 2);
	//cv::Laplacian(img_gray, lp_img, CV_16S, 3);

	//lp_img_u8 = img_gray.clone();
	//lps16_ptr = (short*)lp_img.data;
	//lpu8_ptr = (unsigned char*)lp_img_u8.data;
	//for (i = 0; i < lp_img_u8.rows*lp_img_u8.cols; i++)
	//{
	//	lpu8_ptr[i] = lps16_ptr[i] < 0 ? min(255, -2 * lps16_ptr[i]) : 0;
	//}

#if TEST_COLOR_INVERSE
	img_gray = 255 - img_gray;  //测试反色之后的效果
#endif 

	blobDetector->detect(img_gray, keypoints);

#if TEST_COLOR_INVERSE
	output_img = 255 - img_gray;
#endif 

	calc_hat_feat(output_img.data, &keypoints, output_img.cols, output_img.rows);
	region_group( 255 - img_gray, &keypoints, output_img.cols, output_img.rows, &blob_info );
	

	//Mat img2;
	//cv::resize(img_gray, img2, cv::Size(704, img_bak.rows * 704 / img_bak.cols));
	//bin_img = img2.clone();
	//bernsen_binary(img2.data, bin_img.data, bin_img.cols, bin_img.rows);

	//统计平均亮度
	//int sum_img = 0;
	//for (i = 0; i < bin_img.rows*bin_img.cols;i++)
	//{
	//	sum_img += img2.data[i];
	//}
	//sum_img /= (bin_img.rows*bin_img.cols);
	//for (i = 0; i < bin_img.rows*bin_img.cols; i++)
	//{
		//bin_img.data[i] = 255*(img2.data[i]>sum_img);
	//}


#ifdef PIC_SAVE_PATH
	sprintf(save_filename, "%s/%d_src.jpg", PIC_SAVE_PATH, fram_nr);
#endif 

	/// 画图
	for (iter = keypoints.begin(); iter != keypoints.end(); iter++)
	{
        #if DRAW_FAULT_POINT
        #else 
		if (iter->response)
        #endif
		{
			circle(img_rgb, iter->pt, iter->size, iter->response ? Scalar(0, 0, 255) : Scalar(0, 0, 0), iter->response ? 5 : 1, 8, 0);
		}
	}

	//cv::line(img_rgb, Point(0, img_rgb.rows*SHIELD_HEIGHT_RATIO), Point(img_rgb.cols, img_rgb.rows*SHIELD_HEIGHT_RATIO), CV_RGB(0, 255, 0), 2, 8, 0);

	namedWindow("result", 0);
	imshow("result", img_rgb);

	namedWindow("SRC", 0);
	imshow("SRC", img_bak);

	//namedWindow("LOG", 0);
	//imshow("LOG", lp_img_u8);


	//namedWindow("BIN", 0);
	//imshow("BIN", bin_img);

	
	return 0;
}



static void proc_video( char* filename )
{
	int wait_sec = 0;
	int frame_nr = 0;

	SimpleBlobDetector* detector = NULL;
	vector<KeyPoint> keypoints;
	vector<blob_info_t> blob_info_vec;
#if TEST_VIDEO
	cv::VideoCapture capture(filename);

	if ( capture.isOpened() )
	{

	}
#endif 


	//创建检测器
	create_detector(&detector);

	while (1)
	{
		int key;
		cv::Mat frame;

#if TEST_VIDEO
		capture >> frame;	
#else 
		frame = imread(filename);
#endif 

		if (frame.empty())
		{
			break;
		}


		frame_nr++;

		if (frame_nr % 2 != 0) continue;

		//imwrite("d://1.jpg", frame);

		proc_func(frame, detector, frame_nr, keypoints, blob_info_vec);

		

		key = waitKey(wait_sec);
		printf("key=%d\n", key);

		if (key == 32)
		{
			wait_sec = 1 - wait_sec;
		}

		if (key=='q' || key=='Q')
			break;

	}

	return;
}

#if 0
static void proc_video_match(char* filename)
{
	int wait_sec = 0;
	int frame_nr = 0;
	cv::Mat last_frame;

	SiftFeatureDetector sift_detector;
	SiftDescriptorExtractor sift_feat;
	Mat feat1, feat2;
	
	BFMatcher matcher;
	vector<DMatch> matches;

	
	vector<KeyPoint> kp1, kp2;
	vector<KeyPoint>*p1, *p2;
	vector<KeyPoint>*tmp;

	p1 = &kp1;
	p2 = &kp2;

#if TEST_VIDEO
	cv::VideoCapture capture(filename);

	if (capture.isOpened())
	{

	}
#endif 


	while (1)
	{
		int key;
		cv::Mat frame;
		cv::Mat normal_img;
		
#if TEST_VIDEO
		capture >> frame;
#else 
		frame = imread(filename);
#endif 

		if (frame.empty())
		{
			break;
		}

		frame_nr++;

		if (frame_nr % 2 != 0) continue;

		cv::resize(frame, normal_img, cv::Size( 352,288 ) );
		
		sift_detector.detect(normal_img, *p1);
		sift_feat.compute(normal_img,*p2,feat1);
		matcher.

		if (last_frame.empty())
		{

		}
		else
		{
			sift_detector.detect(last_frame, kp1);
		}
		normal_img.copyTo(last_frame);
	}

}
#endif

static void calc_trans_pt( float* trans_mat,float in_x,float in_y,float* out_x,float* out_y )
{
	
	float xx = trans_mat[0] * in_x + trans_mat[1] * in_y + trans_mat[2];
	float yy = trans_mat[3] * in_x + trans_mat[4] * in_y + trans_mat[5];
	float cc = trans_mat[6] * in_x + trans_mat[7] * in_y + trans_mat[8];

	*out_x = xx / cc;
	*out_y = yy / cc;
}

static void test_homo_show( Mat pimg_infra,
	                        Mat small_img_v,
							float* trans_iv_arr,
							float* trans_vi_arr, 
							int * pt_infra, int *pt_visible,
							int pt_nr,float scale_i,float scale_v )
{
	int i;

	for ( i = 0; i < pt_nr; i++)
	{
		float out_x, out_y;
		calc_trans_pt(trans_iv_arr, pt_infra[2 * i], pt_infra[2 * i + 1], &out_x, &out_y);

		cv::circle(small_img_v, cv::Point(out_x*scale_v, out_y*scale_v), 3, CV_RGB(255, 0, 0), 3, 8, 0);

		calc_trans_pt(trans_vi_arr, pt_visible[2 * i], pt_visible[2 * i + 1], &out_x, &out_y);

		cv::circle(pimg_infra, cv::Point(out_x*scale_i, out_y*scale_i), 3, CV_RGB(255, 0, 0), 3, 8, 0);
	}

	imshow("INFRA", pimg_infra);
	imshow("VISIBLE", small_img_v);
	
}


//根据标定信息，计算变换矩阵
static void  calc_trans_mat( int*pt_infra, int *pt_visible, int pt_nr, float* trans_iv_arr, float* trans_vi_arr )
{
	int i,j;
	CvMat *pt1 = cvCreateMat(pt_nr, 2, CV_32FC1);
	CvMat *pt2 = cvCreateMat(pt_nr, 2, CV_32FC1);
	CvMat *trans_iv = cvCreateMat(3, 3, CV_32FC1);
	CvMat *trans_vi = cvCreateMat(3, 3, CV_32FC1);

	for (i = 0; i < pt_nr; i++)
	{
		cvSet2D(pt1, i, 0, cvScalarAll(pt_infra[2 * i + 0]));
		cvSet2D(pt1, i, 1, cvScalarAll(pt_infra[2 * i + 1]));

		cvSet2D(pt2, i, 0, cvScalarAll(pt_visible[2 * i + 0]));
		cvSet2D(pt2, i, 1, cvScalarAll(pt_visible[2 * i + 1]));
	}

	cvFindHomography(pt1, pt2, trans_iv);
	cvFindHomography(pt2, pt1, trans_vi);


	//测试
	for (i = 0; i < 3; i++)
	{
		for (j = 0; j < 3; j++)
		{
			CvScalar val1 = cvGet2D(trans_iv, i, j);
			CvScalar val2 = cvGet2D(trans_vi, i, j);

			trans_iv_arr[i * 3 + j] = val1.val[0];
			trans_vi_arr[i * 3 + j] = val2.val[0];
		}
	}


	cvReleaseMat( &pt1 );
	cvReleaseMat( &pt2 );
	cvReleaseMat(&trans_iv);
	cvReleaseMat(&trans_vi);

}

/// 视频序列结构体
typedef struct test_quene_info_tag
{
	char* infra_file;             /// 红外图像
	char* visible_file;			  /// 可见光图像
	int infra_delay_nr;			  /// 红外图像预先播放帧数
	int visible_delay_nr;		  /// 可见光图像预先播放帧数
}test_quene_info_t;

static  test_quene_info_t 	test_quene[] = {
	{ "水泥+草地\\5-25红外.AVI",  "水泥+草地\\水泥+草地白光视频.avi", 35, 0 },
	{ "水泥地\\5-30红外.AVI",    "水泥地\\白光水泥地视频.avi",        30, 0 },
	{ "泥土路面\\0-22红外.AVI",  "泥土路面\\泥土路面白光视频.avi",    40, 0 },
	{ "草地\\8-20热成像.AVI",    "草地\\一所草地场景白光1.avi",       0, 40 },
	{ "荒草路面\\5-28红外.AVI",  "荒草路面\\荒草白光视频.avi",        0, 140 }
};


static int print_help()
{
	printf("空格键：暂停/连续运行!!\n");
	printf("v键：   显示/关闭可见光图像!!\n");
	printf("q键：   退出当前视频!!\n");
	return 0;
}


static void test_uva( cv::Mat frame_i )
{
	Mat img_gray;
	Mat blur_big;
	Mat blur_small;

	float sigma_small = 2.5;
	float sigma_big = sigma_small * 5;

	int i;
	int img_w = frame_i.cols;
	int img_h = frame_i.rows;
	Mat img_diff = cv::Mat(img_h, img_w, CV_8UC1);

	unsigned char* ptmp_diff = img_diff.data;
	unsigned char* psmall;
	unsigned char* pbig;
	unsigned char* psrc;

	if (frame_i.channels() > 1)
	{
		cv::cvtColor(frame_i, img_gray, COLOR_RGB2GRAY);
	}
	else
	{
		frame_i.copyTo(img_gray);
	}

	//
	cv::GaussianBlur(img_gray, blur_small, cvSize(3 * sigma_small, 3 * sigma_small), sigma_small, sigma_small);
	cv::GaussianBlur(img_gray, blur_big, cvSize(3 * sigma_big, 3 * sigma_big), sigma_big, sigma_big);

	psmall = blur_small.data;
	pbig = blur_big.data;
	psrc = img_gray.data;

	for (i = 0; i<img_h*img_w;i++)
	{
		ptmp_diff[i] = (psmall[i] >(pbig[i] + 10))*(psmall[i]);// -pbig[i]);
	}
	//img_diff = 255 * ((blur_small - blur_big) > 30 & (blur_small>100));

	cv::imshow("blur_small", blur_small);
	cv::imshow("blur_big", blur_big);
	cv::imshow("img_diff", img_diff);
}

static int test_two_video(char* file_path, char* model_path)
{
	int k,select_i;

	char fn_infra[260];
	char fn_visible[260];
	//char file_path[260] = "F:\\data\\工程兵扫雷项目\\2016年（冬）场景记录表\\";	   //目录

	char lbp_svm_model_fn[260];
	char hog_svm_model_fn[260];


	//标定信息
	int pt_infra[]   = { 420, 480, 577, 529, 621, 446, 714,482,         482,112,647,115,812,106,923,77      ,601,722,865,602,745,96,916,96 };	   //荒草路面
	int pt_visible[] = { 878, 650, 1062, 707, 1111, 608, 1213, 652,   943,225,1136,228,1327,215,1453,196  ,1103,933,1403,797,1252,209,1449,209  };

	int pt_nr = (sizeof(pt_infra) / sizeof(pt_infra[0]))>>1;

	int infra_delay_nr   = 0;    // 红外图像延迟帧数
	int visible_delay_nr = 0;  //可见光图像延迟帧数
	int frame_i_nr = 0;
	int frame_v_nr = 0;
	int sec = 1;
	int save_img = 0;               //是否保存图片
	int show_visible_img = 0;		//是否显示可见光图片


	//打印文件列表
	for (k = 0; k < sizeof(test_quene) / sizeof(test_quene[0]);k++)
	{
		printf("%d:%s\n", k, test_quene[k].infra_file);
	}

	printf("select:");
	scanf("%d", &select_i);
	if ( select_i>=0 && select_i < (sizeof(test_quene) / sizeof(test_quene[0])))
	{
		sprintf(fn_infra, "%s//%s", file_path, test_quene[select_i].infra_file);
		sprintf(fn_visible, "%s//%s", file_path, test_quene[select_i].visible_file);
		infra_delay_nr = test_quene[select_i].infra_delay_nr;
		visible_delay_nr = test_quene[select_i].visible_delay_nr;
	}
	else
	{
		printf("filename:");
		scanf("%s", fn_infra);
		infra_delay_nr = visible_delay_nr = 0;
		visible_delay_nr = -1;
		fn_visible[0] = 0;
	}


	cv::VideoCapture cap_i(fn_infra);	
	cv::VideoCapture cap_v(fn_visible);

	//SimpleBlobDetector* detector = NULL;
	vector<KeyPoint> keypoints;
	vector<blob_info_t> blob_info_vec;
	vector<KeyPoint>::iterator iter;
	vector<blob_info_t>::iterator iter2;
	blob_detect_params_t detect_param;

	
	Mat lbp_map;
	Mat grad_mag, grad_ang;

	//变换矩阵
	float trans_iv_arr[9];
	float trans_vi_arr[9];

	//lbp和hog模型
	int hist_bin_nr = 59;
	int counter = 0;
	char save_filename[260];
	struct svm_model* svm_model_lbp = NULL;
	struct svm_node * data_node_lbp = (struct svm_node *)malloc(sizeof(struct svm_node) * ( hist_bin_nr * 10 + 1 ));
	float *phist_lbp = new float[hist_bin_nr * 10];

	struct svm_model* svm_model_hog = NULL;
	struct svm_node * data_node_hog = (struct svm_node *)malloc(sizeof(struct svm_node) * ( 8 * 10 +1 ) );
	float *phist_hog = new float[8 * 10];
	char title[260];


	//检测参数
	memset(&detect_param, 0, sizeof(detect_param));
	detect_param.blur_size = 0;
	detect_param.min_area = 5 * 5;
	detect_param.max_area = 25 * 25;
	detect_param.region_grow_thr = 30;
	detect_param.centre_gray_thr = 100;
	detect_param.shield_height_ratio = 0;

	//
	sprintf(lbp_svm_model_fn, "%s//model1.dat", model_path);
	sprintf(hog_svm_model_fn, "%s//model2.dat", model_path);


	print_help();


	if (!cap_i.isOpened())
	{
		printf("open file %s failed!!\n", fn_infra);
		return -1;
	}

	if (visible_delay_nr>=0 && !cap_v.isOpened())
	{
		printf("open file %s failed!!\n", fn_visible);
		return -1;
	}

#if 0
	frame_v_nr = 0;
	do 
	{
		Mat frame_i, frame_v;
		cap_i >> frame_i;
		cap_v >> frame_v;

		if (frame_i.empty())break;
		if (frame_v.empty())break;

		frame_v_nr++;
		sprintf(title, "%d", frame_v_nr);

		cv::putText(frame_v, title, Point(20, frame_v.rows - 20), CV_FONT_HERSHEY_SIMPLEX, 1, CV_RGB(255, 0, 0));

		namedWindow("VISIBLE", 0);
		namedWindow("INFRA", 0);
		imshow("VISIBLE", frame_v);
		imshow("INFRA", frame_i);

		int key = cvWaitKey(sec);
		if (key == ' ')
		{
			sec = 1 - sec;
		}

		if (key == 's')
		{
			save_img = 1 - save_img;
		}

	} while (1);

	return 0;
#endif 


	//跳帧处理
	if (visible_delay_nr>0)
		cap_v.set(CV_CAP_PROP_POS_FRAMES, visible_delay_nr);

	if (infra_delay_nr>0)
		cap_i.set(CV_CAP_PROP_POS_FRAMES, infra_delay_nr);

	frame_v_nr = visible_delay_nr;
	frame_i_nr = 0;// infra_delay_nr;


	//创建检测器
	//create_detector(&detector);

	//计算可见光和红外图像的映射变换矩阵
	calc_trans_mat( pt_infra, pt_visible, pt_nr,trans_iv_arr,trans_vi_arr );


	//加载模型
	svm_model_lbp = svm_load_model(lbp_svm_model_fn);
	svm_model_hog = svm_load_model(hog_svm_model_fn);

	if (svm_model_lbp == NULL || svm_model_hog == NULL)
	{
		goto FUNC_EXIT;
	}

	//可见光25fps,热成像7.5fps
	do 
	{
		Mat frame_i, frame_v;
		Mat frame_i_back;

		if (visible_delay_nr >= 0)
		{
			cap_v >> frame_v;
			if (frame_v.empty())
				break;
		}

		frame_v_nr++;

		if (frame_v_nr % 3 == 0)
		{
			frame_i_nr++;
			if ((frame_i_nr % 15) == 0)
			{
				continue;
			}

			cap_i >> frame_i;
			if (frame_i.empty())
				break;
		}
		else
		{
			continue;
		}

		if ((frame_i_nr % 2) == 0) continue;

		//利用热成像计算可疑点
		//keypoints.clear();
		if (frame_v.empty())
			frame_i.copyTo(frame_v);

		frame_i.copyTo(frame_i_back);
		cv::GaussianBlur(frame_i, frame_i, cv::Size(3, 3), 2 );
		
		test_uva(frame_i_back);
		//获取第一级的结果
		//proc_func(frame_i, detector, frame_i_nr, keypoints, blob_info_vec);
		blob_detect_process(frame_i, &detect_param, &blob_info_vec);

	
		Mat img_gray;
		Mat img_gray_infra;
		cv::cvtColor(frame_v, img_gray, COLOR_RGB2GRAY);
		cv::cvtColor(frame_i, img_gray_infra, COLOR_RGB2GRAY);

		if (lbp_map.empty())
		{
			lbp_map = Mat::zeros(cv::Size(frame_v.cols, frame_v.rows), CV_8UC1);
		}

		if (grad_mag.empty())
		{
			grad_mag = Mat::zeros(cv::Size(frame_i.cols, frame_i.rows), CV_8UC1);
		}

		if (grad_ang.empty())
		{
			grad_ang = Mat::zeros(cv::Size(frame_i.cols, frame_i.rows), CV_8UC1);
		}

		//

		calc_lbp_map(img_gray.data, lbp_map.data, lbp_map.rows, lbp_map.cols);
		//
		calc_hog_gradient_img(img_gray_infra.data, grad_mag.data, grad_ang.data, img_gray_infra.rows, img_gray_infra.cols);

#if 1

		for (iter2 = blob_info_vec.begin(); iter2 != blob_info_vec.end(); iter2++)
		{
			blob_info_t blob = *iter2;
			int label_v = 1,label_i = 1;
			rect_t roi_map_rect;
			int feat_dim = 0;
			int feat_dim2 = 0;
			float out_xs[2], out_ys[2];
			double prob = 1.0f;

			calc_trans_pt(trans_iv_arr, blob.bd_rect.left, blob.bd_rect.top, &out_xs[0], &out_ys[0]);
			calc_trans_pt(trans_iv_arr, blob.bd_rect.right, blob.bd_rect.bottom, &out_xs[1], &out_ys[1]);

#if 0
			//预测
			roi_map_rect.left = out_xs[0];
			roi_map_rect.right = out_xs[1];
			roi_map_rect.top = out_ys[0];
			roi_map_rect.bottom = out_ys[1];
			feat_dim = calc_lbp_feat_roi(lbp_map.data, img_gray.cols, phist, hist_bin_nr, &roi_map_rect);

			for (k = 0; k < feat_dim;k++)
			{
				data_node[k].index = k + 1;
				data_node[k].value = phist[k];
			}
			data_node[k].index = -1;
			label = svm_predict_probability(svm_model_ptr, data_node, &prob);
#endif 

#if 1
			feat_dim2 = calc_hog_feat_roi(grad_mag.data, grad_ang.data, grad_mag.cols, phist_hog, 8, &blob.bd_rect);
			
			for (k = 0; k < feat_dim2; k++)
			{
				data_node_hog[k].index = k + 1;
				data_node_hog[k].value = phist_hog[k];
			}
			data_node_hog[k].index = -1;
			//printf("1111--11\n");
			//label_i = svm_predict_probability(svm_model_hog, data_node_hog, &prob);
			label_i = svm_predict(svm_model_hog, data_node_hog);
#endif 
			
			//图像保存
			//Mat roi_mat = cur_mask(Rect(rect_roi_sub.left, rect_roi_sub.top, rect_roi_sub.right - rect_roi_sub.left + 1, rect_roi_sub.bottom - rect_roi_sub.top + 1));

#ifdef _SIMPLE_COLLOECT_MODE
			if (save_img )
			{
				Mat roi_img = frame_v(Rect(out_xs[0], out_ys[0], out_xs[1] - out_xs[0] + 1, out_ys[1] - out_ys[0] + 1));

				if ((blob.bd_rect.right - blob.bd_rect.left) > 15 && (blob.bd_rect.bottom - blob.bd_rect.top) > 15)
				{
					Mat roi_img_infra = frame_i_back(Rect(blob.bd_rect.left, blob.bd_rect.top, blob.bd_rect.right - blob.bd_rect.left + 1, blob.bd_rect.bottom - blob.bd_rect.top + 1));

					//sprintf(save_filename, "f://data//landmine//%d.jpg", counter);
					//imwrite(save_filename, roi_img);
					sprintf(save_filename, "f://data//landmine//infra_%d.jpg", counter);
					imwrite(save_filename, roi_img_infra);
				}
			}
#endif
				
			counter++;

			if (label_i > 0)
			{
				cv::rectangle(frame_v, cv::Point(out_xs[0], out_ys[0]), cv::Point(out_xs[1], out_ys[1]), CV_RGB(255, 0, 0), 3, 8, 0);
				cv::rectangle(frame_i, cv::Point(blob.bd_rect.left, blob.bd_rect.top), cv::Point(blob.bd_rect.right, blob.bd_rect.bottom), CV_RGB(255, 0, 0), 3, 8, 0);
				
			}
			else
			{
				//cv::rectangle(frame_v, cv::Point(out_xs[0], out_ys[0]), cv::Point(out_xs[1], out_ys[1]), CV_RGB(0, 0, 0), 3, 8, 0);
				//cv::rectangle(frame_i_bak, cv::Point(blob.bd_rect.left, blob.bd_rect.top), cv::Point(blob.bd_rect.right, blob.bd_rect.bottom), CV_RGB(0, 0, 0), 3, 8, 0);
			}
		}

		//namedWindow("LBP", 0);
		//imshow("LBP", lbp_map);

#ifdef _DEBUG_MODE 
		namedWindow("GRAD", 0);
		imshow("GRAD", grad_mag);
#endif 

		//显示
		sprintf(title, "%d", frame_v_nr);

		cv::putText(frame_v, title, Point(20, frame_v.rows - 20), CV_FONT_HERSHEY_SIMPLEX, 1, CV_RGB(255, 0, 0));

#endif 

		if (show_visible_img && visible_delay_nr>=0 )
		{
			namedWindow("VISIBLE", 0);
			imshow("VISIBLE", frame_v);
		}
		else
		{
			cvDestroyWindow("VISIBLE");
		}

		namedWindow("INFRA", 0);
		imshow("INFRA", frame_i);
		//imwrite("d://1.jpg",frame_v);
		//imwrite("d://2.jpg",frame_i_bak);

		int key = cvWaitKey(sec);
		if (key == ' ')
		{
			sec = 1 - sec;
		}
		else if (key == 's')
		{
			save_img = 1 - save_img;
		}
		else if (key == 'v')
		{
			show_visible_img = 1 - show_visible_img;
		}
		else if (key == 'q')
		{
			break;
		}

	} while (1);


FUNC_EXIT:

    cvDestroyAllWindows();
	if (svm_model_lbp)svm_free_and_destroy_model(&svm_model_lbp);
	if (data_node_lbp)free(data_node_lbp);
	if (phist_lbp) delete[] phist_lbp;


	if (svm_model_hog) svm_free_and_destroy_model(&svm_model_hog);
	if (data_node_hog) free(data_node_hog);
	if (phist_hog) delete[] phist_hog;

	return 0;
}


static int test_homo()
{
	int i,j;
	//char fn1[260] = "Z:\\工程兵探雷\\2016年（冬）场景记录表\\荒草路面\\红外抠图.jpg";
	//char fn2[260] = "Z:\\工程兵探雷\\2016年（冬）场景记录表\\荒草路面\\荒草路面白光.png";

	char fn1[260] = "F:\\data\\工程兵扫雷项目\\2016年（冬）场景记录表\\荒草路面\\荒草路面白光.png";
	char fn2[260] = "F:\\data\\工程兵扫雷项目\\2016年（冬）场景记录表\\荒草路面\\荒草路面白光.png";

	int pt_nr = 4;

	//int pt_infra[] = {288,285,390,312,478,287,289,328};
	int pt_infra[]   = { 420,480,577, 529, 621,446, 714,482 };	   //荒草路面
	int pt_visible[] = { 878,650,1062,707, 1111,608,1213,652 };
	//int pt_infra[] = { 579,288,643,268,642,359,693,327 };	   //水泥+草地
	//int pt_visible[] = {1059,420,1129,399,1128,503,1194,468 };

	CvMat *pt1 = cvCreateMat( pt_nr,2,CV_32FC1 );
	CvMat *pt2 = cvCreateMat(pt_nr, 2, CV_32FC1);
	CvMat *trans_iv = cvCreateMat(3, 3, CV_32FC1);
	CvMat *trans_vi = cvCreateMat(3, 3, CV_32FC1);

	Mat pimg_infra = imread(fn1);
	Mat pimg_v = imread(fn2);
	Mat small_img_v;// = Mat(pimg_v.rows() / 2, pimg_v.cols() / 2, pimg_v.type()）;

	cv::resize(pimg_v, small_img_v, cv::Size(pimg_v.cols / 2, pimg_v.rows/2));

	float trans_iv_arr[9];
	float trans_vi_arr[9];

	for (i = 0; i < pt_nr;i++)
	{
		cvSet2D(pt1, i, 0, cvScalarAll(pt_infra[2*i+0]));
		cvSet2D(pt1, i, 1, cvScalarAll(pt_infra[2*i+1]));

		cvSet2D(pt2, i, 0, cvScalarAll(pt_visible[2 * i + 0]));
		cvSet2D(pt2, i, 1, cvScalarAll(pt_visible[2 * i + 1]));
	}

	cvFindHomography( pt1, pt2, trans_iv );
	cvFindHomography( pt2, pt1, trans_vi );


	//测试
	for (i = 0; i < 3;i++)
	{
		for (j = 0; j < 3;j++)
	   {
			CvScalar val1 = cvGet2D(trans_iv,i,j);
			CvScalar val2 = cvGet2D(trans_vi, i, j);

			trans_iv_arr[i * 3 + j] = val1.val[0];
			trans_vi_arr[i * 3 + j] = val2.val[0];
	   }
	}


	int frame_nr_v = 0;
	int frame_nr_i = 0;
	int sec = 0;
	int step_i = 0;
	//cv::VideoCapture cap_i("Z:\\工程兵探雷\\2016年（冬）场景记录表\\荒草路面\\5-28红外2.AVI");
	//cv::VideoCapture cap_v("Z:\\工程兵探雷\\2016年（冬）场景记录表\\荒草路面\\荒草白光视频.avi");

	//cv::VideoCapture cap_i("F:\\data\\工程兵扫雷项目\\2016年（冬）场景记录表\\荒草路面\\5-28红外3.AVI");	    //红外相差144帧
	//cv::VideoCapture cap_v("F:\\data\\工程兵扫雷项目\\2016年（冬）场景记录表\\荒草路面\\荒草白光视频.avi");

	//cv::VideoCapture cap_i("F:\\data\\工程兵扫雷项目\\2016年（冬）场景记录表\\草地\\8-20热成像.AVI");	    //红外相差24帧
	//cv::VideoCapture cap_v("F:\\data\\工程兵扫雷项目\\2016年（冬）场景记录表\\草地\\一所草地场景白光1.avi");

	cv::VideoCapture cap_i("F:\\data\\工程兵扫雷项目\\2016年（冬）场景记录表\\泥土路面\\0-22红外.AVI");	    //红外图像早120帧左右
	cv::VideoCapture cap_v("F:\\data\\工程兵扫雷项目\\2016年（冬）场景记录表\\泥土路面\\泥土路面白光视频.avi");
	

	if (!cap_i.isOpened())
	{
		printf("open avi failed!!!\n");
		return 0;
	}
	
	if (!cap_v.isOpened())
	{
		printf("open avi failed!!!\n");
		return 0;
	}

	int throw_nr = 0;
	do 
	{
		Mat frame_i,frame_v;
		Mat small_i,small_v;

		if (step_i == 0)
		{
			cap_v >> frame_v;
		}
		frame_nr_v++;

		if ( ( (frame_nr_v) % 3 ) == 0 &&  frame_nr_v>0)//144)
		{

			if ((frame_nr_i % 15) == 1) { frame_nr_i++; continue; }


		INFRA_STEP:
			//printf("11111\n");
			cap_i >> frame_i;
			frame_nr_i++;
			if (frame_i.empty()) break;

		}



		if (frame_v.empty())break;

		//imwrite("D:\\infra.jpg", frame_i);
		//imwrite("D:\\visible.jpg", frame_v);

		//if ( frame_nr_i  % 3 != 0) continue;

		if (!frame_i.empty())
		{
			char title[260];
			cv::resize(frame_v, small_v, cv::Size(frame_v.cols / 2, frame_v.rows / 2));
			cv::resize(frame_i, small_i, cv::Size(frame_i.cols / 2, frame_i.rows / 2));

			sprintf(title, "%d", frame_nr_v);
			cv::putText(small_v, title, Point(20, small_v.rows-20), CV_FONT_HERSHEY_SIMPLEX, 1, CV_RGB(255, 0, 0));
			test_homo_show(small_i, small_v, trans_iv_arr, trans_vi_arr, pt_infra, pt_visible, pt_nr, 0.5, 0.5);

			int key = cvWaitKey(sec);
			if (key == ' ')
			{
				sec = 1 - sec;
			}

			if (key == 's')
			{
				step_i = 1;
				goto INFRA_STEP;
			}
			if (key == 'q')
			{
				step_i = 0;
			}
		}

	} while (1);

	test_homo_show(pimg_infra, small_img_v, trans_iv_arr, trans_vi_arr, pt_infra, pt_visible, pt_nr,1,0.5);
	cvWaitKey(0);

	cvReleaseMat(&pt1);
	cvReleaseMat(&pt2);
	cvReleaseMat(&trans_iv);
	cvReleaseMat(&trans_vi);

	return 0;

}



#define UNKNOWN_FLOW_THRESH 1e9  

void makecolorwheel(vector<Scalar> &colorwheel)
{
	int RY = 15;
	int YG = 6;
	int GC = 4;
	int CB = 11;
	int BM = 13;
	int MR = 6;

	int i;

	for (i = 0; i < RY; i++) colorwheel.push_back(Scalar(255, 255 * i / RY, 0));
	for (i = 0; i < YG; i++) colorwheel.push_back(Scalar(255 - 255 * i / YG, 255, 0));
	for (i = 0; i < GC; i++) colorwheel.push_back(Scalar(0, 255, 255 * i / GC));
	for (i = 0; i < CB; i++) colorwheel.push_back(Scalar(0, 255 - 255 * i / CB, 255));
	for (i = 0; i < BM; i++) colorwheel.push_back(Scalar(255 * i / BM, 0, 255));
	for (i = 0; i < MR; i++) colorwheel.push_back(Scalar(255, 0, 255 - 255 * i / MR));
}

void motionToColor(Mat flow, Mat &color)
{
	if (color.empty())
		color.create(flow.rows, flow.cols, CV_8UC3);

	static vector<Scalar> colorwheel; //Scalar r,g,b  
	if (colorwheel.empty())
		makecolorwheel(colorwheel);

	// determine motion range:  
	float maxrad = -1;

	// Find max flow to normalize fx and fy  
	for (int i = 0; i < flow.rows; ++i)
	{
		for (int j = 0; j < flow.cols; ++j)
		{
			Vec2f flow_at_point = flow.at<Vec2f>(i, j);
			float fx = flow_at_point[0];
			float fy = flow_at_point[1];
			if ((fabs(fx) >  UNKNOWN_FLOW_THRESH) || (fabs(fy) >  UNKNOWN_FLOW_THRESH))
				continue;
			float rad = sqrt(fx * fx + fy * fy);
			maxrad = maxrad > rad ? maxrad : rad;
		}
	}

	for (int i = 0; i < flow.rows; ++i)
	{
		for (int j = 0; j < flow.cols; ++j)
		{
			uchar *data = color.data + color.step[0] * i + color.step[1] * j;
			Vec2f flow_at_point = flow.at<Vec2f>(i, j);

			float fx = flow_at_point[0] / maxrad;
			float fy = flow_at_point[1] / maxrad;
			if ((fabs(fx) >  UNKNOWN_FLOW_THRESH) || (fabs(fy) >  UNKNOWN_FLOW_THRESH))
			{
				data[0] = data[1] = data[2] = 0;
				continue;
			}
			float rad = sqrt(fx * fx + fy * fy);

			float angle = atan2(-fy, -fx) / CV_PI;
			float fk = (angle + 1.0) / 2.0 * (colorwheel.size() - 1);
			int k0 = (int)fk;
			int k1 = (k0 + 1) % colorwheel.size();
			float f = fk - k0;
			//f = 0; // uncomment to see original color wheel  

			for (int b = 0; b < 3; b++)
			{
				float col0 = colorwheel[k0][b] / 255.0;
				float col1 = colorwheel[k1][b] / 255.0;
				float col = (1 - f) * col0 + f * col1;
				if (rad <= 1)
					col = 1 - rad * (1 - col); // increase saturation with radius  
				else
					col *= .75; // out of range  
				data[2 - b] = (int)(255.0 * col);
			}
		}
	}
}

static int  calc_optical_flow(Mat prevgray, Mat gray)
{
	Mat flow;
	Mat motion2color;

	cv::cvtColor(prevgray, prevgray, CV_RGB2GRAY);
	cv::cvtColor(gray, gray, CV_RGB2GRAY);

	cv::resize(prevgray, prevgray, Size(352, 288));
	cv::resize(gray, gray, Size(352, 288));

	calcOpticalFlowFarneback(prevgray, gray, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
	motionToColor(flow, motion2color);
	imshow("flow", motion2color);
	return 0;
}



static int test_uav_detect(int visible_img=0 )
{
	int max_nr = 1;
	char fn_infra[260];
	cv::Mat frame;
	cv::Mat last_frame;

	vec_rect_t ret;
	int frame_nr = 0;
	int wait_sec = 1;
	void* track_handle;

	printf("filename:");
	scanf("%s", fn_infra);

	tracker_create(&track_handle);


	cv::VideoCapture cap_i(fn_infra);

	if (!cap_i.isOpened())
	{
		printf("open file %s failed!!\n", fn_infra);
		return -1;
	}

	do
	{
		int k;
		int wait_key;
		char title[32];
		vec_rect_t::iterator iter;
		cap_i >> frame;
		if (frame.empty())
		{
			break;
		}

		frame_nr++;

		if ((frame_nr % 3) != 0) continue;
		//主处理
		//uav_detect_process(visible_img?( cv::Scalar(255,255,255)-frame):frame, &ret);

		//if (!last_frame.empty())
		//{
		//	calc_optical_flow(last_frame, frame);
		//}
		tracker_process(track_handle, frame);

		frame.copyTo(last_frame);

		k = 0;
		for (iter = ret.begin(); iter != ret.end();iter++)
		{
			uav_rect_t rect = *iter;
			cv::rectangle(frame, cv::Rect(cv::Point(rect.left, rect.top), cv::Point(rect.right, rect.bottom)), cv::Scalar(0, 255, 0, 0), 3);

			sprintf(title, "id:%d %.3f", k,rect.score / 10000.f);
			cv::putText( frame, title, 
				         Point( (rect.right<frame.cols - 50) ? rect.right:rect.left-25, 
						        (rect.top>50)?rect.top : (rect.bottom +15 )), 
					     cv::FONT_HERSHEY_PLAIN, 
						 2.0, Scalar(0, 0, 255), 3, 8);

			k++;
			if (k >= max_nr)
				break;
		}

		sprintf(title, "%d", frame_nr);
		cv::putText(frame, title, cv::Point(15, 30), cv::FONT_HERSHEY_PLAIN, 2.0, Scalar(0, 0, 255, 0),3);

		cv::namedWindow("RET", 0);
		cv::imshow("RET", frame);

		wait_key = cv::waitKey(wait_sec);
		if (wait_key == ' ')
		{
			wait_sec = 1 - wait_sec;
		}
		else if (wait_key == 'q' || wait_key == 'Q')
		{
			break;
		}

	} while (1);

	cv::destroyAllWindows();

	return 0;
}

int main(int argc, char** argv)  
{
	char filename[260] = {"E:\\HIKDATA\\data\\2016年冬场景记录表\\草地\\马上试验场红外区域.jpg"};
	int is_visile = 0;
	if (argc > 1)
	{
		is_visile = atoi(argv[1]);
	}
//	test_homo();
#if 1
	do 
	{
		test_uav_detect(is_visile);
	} while (1);
	
	//generate_svm_train_dat( "F:\\data\\infra\\", "F:\\data\\infra\\simple_hog2.dat");
#else 
	if (argc >= 3)
	{
		do 
		{
			test_two_video(argv[1], argv[2]);

		} while (1);
		
	}
	else
	{
		printf("argument is error!!!");
	}
#endif 

	return 0;

#if 0
	do 
	{
		printf("file:");
		scanf("%s",filename);
		printf("input=%s\n", filename);

		//blob_detect(filename);
		//proc_video(filename);
		//fun(filename);

		printf("endl\n");
	} while (1);
#endif 
}

#endif 