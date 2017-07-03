#include "type.h"
#include "blob_detect.h"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <vector>

using namespace cv;


//计算积分图
static void calc_integral_img(unsigned char* pimg, int* pintegral, int img_w, int img_h)
{
	int i, j;
	int* ptmp_int = pintegral;
	pintegral[0] = pimg[0];
	for (i = 1; i < img_w; i++)
	{
		pintegral[i] = pimg[i] + pintegral[i - 1];
	}

	pimg += img_w;
	ptmp_int += img_w;
	for (i = 1; i < img_h; i++)
	{
		int sum = 0;
		for (j = 0; j < img_w; j++)
		{
			sum += pimg[j];
			ptmp_int[j] = pintegral[j] + sum;
		}

		ptmp_int += img_w;
		pintegral += img_w;
		pimg += img_w;
	}
}


//计算积分图的值
static int calc_block_sum(int* pintegral, int left, int right, int top, int bottom, int img_w)
{
	int val = pintegral[top*img_w + left] + pintegral[bottom*img_w + right] - pintegral[top*img_w + right] - pintegral[bottom*img_w + left];
	return val / ((right - left + 1)*(bottom - top + 1));
}


//计算候选点附近的对比特性
static void calc_hat_feat( unsigned char* pimg, vector<KeyPoint>* keypoints, int img_w, int img_h )
{
	int k;
	vector<KeyPoint>::iterator iter;
	int* pintegral = new int[img_h*img_w];
	float bd_plus = 2;

	calc_integral_img(pimg, pintegral, img_w, img_h);

	for (iter = keypoints->begin(); iter != keypoints->end(); iter++)
	{
		int thr = 15;
		int x = iter->pt.x;
		int y = iter->pt.y;
		float size = MAX(5, iter->size);
		int left, top, right, bottom;
		int radius = MAX(7, MIN(15, size*bd_plus));
		int val[8];

		iter->response = 0;

		if ((x + (radius + size)) >= (img_w - 3) ||
			(y + (radius + size)) >= (img_h - 3) ||
			(x - (radius + size)) <= 3 ||
			(y - (radius + size)) <= 3)
		{
			continue;
		}

		//if (y < img_h*SHIELD_HEIGHT_RATIO)
		//{
		//	continue;
		//}

		//计算模块均值
		left = x - size;
		right = x + size;
		top = y - size;
		bottom = y + size;

		val[0] = calc_block_sum(pintegral, left - radius, left, top - radius, top, img_w);
		val[1] = calc_block_sum(pintegral, left, right, top - radius, top, img_w);
		val[2] = calc_block_sum(pintegral, right, right + radius, top - radius, top, img_w);

		val[3] = calc_block_sum(pintegral, left - radius, left, top, bottom, img_w);
		int v11 = calc_block_sum(pintegral, left, right, top, bottom, img_w);
		val[4] = calc_block_sum(pintegral, right, right + radius, top, bottom, img_w);

		val[5] = calc_block_sum(pintegral, left - radius, left, bottom, bottom + radius, img_w);
		val[6] = calc_block_sum(pintegral, left, right, bottom, bottom + radius, img_w);
		val[7] = calc_block_sum(pintegral, right, right + radius, bottom, bottom + radius, img_w);

		//比较阈值
		thr = MAX((v11*0.15), 20);
		int thr_low = MIN((v11*0.1), 10);

		int big_nr = 0;//
		int big_nr_low = 0;//

		for (k = 0; k<8; k++)
		{
			big_nr += v11>(val[k] + thr);
			big_nr_low += v11 > (val[k] + thr_low);
		}

		if (big_nr_low >= 8 && big_nr >= 1)
		{
			iter->response = 1;
		}

	}

	delete[] pintegral;
}


//区域生长,返回生长掩码图和区域的均值和面积
static  Mat seed_region_group(unsigned char* pimg, KeyPoint* key_pt, rect_t* rect, int img_w, int img_h, int th, int *region_area, int *region_avg_val)
{
	int area = 0;
	int avg_val = 0;
	Point2i pt(key_pt->pt.x, key_pt->pt.y);
	Point2i ptGrowing;						//待生长点位置
	int nGrowLable = 0;								//标记是否生长过
	int nSrcValue = 0;								//生长起点灰度值
	int nCurValue = 0;								//当前生长点灰度值
	Mat matDst = Mat::zeros(cv::Size(img_w, img_h), CV_8UC1);	//创建一个空白区域，填充为黑色

	//生长方向顺序数据
	int DIR[8][2] = { { -1, -1 }, { 0, -1 }, { 1, -1 }, { 1, 0 }, { 1, 1 }, { 0, 1 }, { -1, 1 }, { -1, 0 } };
	Vector<Point2i> vcGrowPt;						//生长点栈
	vcGrowPt.push_back(pt);							//将生长点压入栈中
	matDst.at<uchar>(pt.y, pt.x) = 255;				//标记生长点
	nSrcValue = pimg[pt.y*img_w + pt.x];			//记录生长点的灰度值

	th = MAX( th,nSrcValue*0.25);

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
			if (ptGrowing.x < rect->left || ptGrowing.y < rect->top || ptGrowing.x > rect->right || ptGrowing.y > rect->bottom)
				continue;

			nGrowLable = matDst.at<uchar>(ptGrowing.y, ptGrowing.x);		//当前待生长点的灰度值

			if (nGrowLable == 0)					//如果标记点还没有被生长
			{
				nCurValue = pimg[ptGrowing.y*img_w + ptGrowing.x];
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


 //中心点定义
typedef struct  Center_tag
{
	Point2d location;
	double radius;
	double confidence;
	blob_info_t info;
}Center;

static void findBlobs(const cv::Mat &binaryImage, vector<Center> &centers)
{
	vector < vector<Point> > contours;
	Mat tmpBinaryImage = binaryImage.clone();
	centers.clear();

	findContours(tmpBinaryImage, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

	//if ( contours.size() > 1) return;

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

		if (moms.m00<4*4) continue;

		//面积
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

		if (1)//params.filterByConvexity)
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

		if (0) //params.filterByColor)
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
				rect_roi.left = MIN(rect_roi.left, pt.x);
				rect_roi.right = MAX(rect_roi.right, pt.x);
				rect_roi.top = MIN(rect_roi.top, pt.y);
				rect_roi.bottom = MAX(rect_roi.bottom, pt.y);
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

//
static void calc_roi_area_and_avg_by_mask( unsigned char* pimg,unsigned char* pmask,rect_t* prect,int img_w,int *parea,int *pavg_val )
{
	int i, j;
	int area = 1;
	int sum = 0;
	//unsigned char* pimg_tmp, unsigned char* pmask

	pimg += prect->top * img_w;
	pmask += prect->top * img_w;
	for (i = prect->top; i <= prect->bottom;i++)
	{
		for (j = prect->left; j <= prect->right; j++)
		{
			sum += (pmask[j]>0)*pimg[j];
			area += (pmask[j]>0);
		}

		pimg += img_w;
		pmask += img_w;
	}

	*parea = area;
	*pavg_val = sum / area;
}

//计算全图的均值




//计算候选点附近的对比特性
static void region_group(Mat gray_img, blob_detect_params_t*detect_params, vector<KeyPoint>* keypoints, int img_w, int img_h, vector<blob_info_t>* blob_info_vec)
{
	int i, j;
	vector<KeyPoint>::iterator iter;
	vector<KeyPoint> bak_pts;
	vector<Center>::iterator iter2;
	unsigned char* pimg = gray_img.data;

	Mat matDst = Mat::zeros(cv::Size(img_w, img_h), CV_8UC1);	//创建一个空白区域，填充为黑色
	int thr = detect_params->region_grow_thr;
	char filename[260];

	if (blob_info_vec)
	{
		blob_info_vec->clear();
	}

	//遍历所有的候选点
	for (iter = keypoints->begin(); iter != keypoints->end(); iter++)
	{
		KeyPoint key_pt = *iter;
		rect_t rect_roi, rect_roi_sub;

		int k = 5;
		int area, avg_val;

		//如果无效的点不考虑
		if (key_pt.response == 0) continue;
		//key_pt.response = 0;

		rect_roi.left = MAX(0, key_pt.pt.x - k * key_pt.size);
		rect_roi.right = MIN(img_w - 1, key_pt.pt.x + k * key_pt.size);
		rect_roi.top = MAX(0, key_pt.pt.y - k * key_pt.size);
		rect_roi.bottom = MIN(img_h - 1, key_pt.pt.y + k * key_pt.size);

		//区域生长
		Mat cur_mask = seed_region_group(pimg, &key_pt, &rect_roi, img_w, img_h, thr, &area, &avg_val);

		//得到局部小图,然后对小图进行findcounter操作
		//Mat roi_mat = cur_mask(Rect(rect_roi.left, rect_roi.top, rect_roi.right - rect_roi.left, rect_roi.bottom - rect_roi.top));

		matDst |= cur_mask;
	}

	vector<Center> centers;
	Mat kernel = cv::getStructuringElement( cv::MORPH_RECT, cv::Size(5, 5),cv::Point(2,2) );

	//膨胀
	cv::dilate(matDst, matDst,kernel );
	cv::erode(matDst, matDst, kernel);
	findBlobs(matDst, centers);
	
	for (iter2 = centers.begin(); iter2 != centers.end();iter2++)
	{
		Center center = *iter2;
		KeyPoint key_pt;
		int region_area,avg_val;
		int rect_w = center.info.bd_rect.right - center.info.bd_rect.left + 1;
		int rect_h = center.info.bd_rect.bottom - center.info.bd_rect.top + 1;
		int is_valid = 1;

		key_pt.pt.x = ( center.info.bd_rect.left + center.info.bd_rect.right )/2;
		key_pt.pt.y = ( center.info.bd_rect.top + center.info.bd_rect.bottom)/ 2;

		//图像上半部分过滤掉
		if (key_pt.pt.y < gray_img.rows* detect_params->shield_height_ratio )
			continue;

		//计算区域的累积值
#if 1
		for (iter = keypoints->begin(); iter != keypoints->end(); iter++)
		{
			KeyPoint key_pt = *iter;
			int val_label = matDst.at<uchar>(key_pt.pt.y, key_pt.pt.x);

			if (val_label)
			{
				if ( rect_w > 3 * 2 * key_pt.size ||
					 rect_h > 3 * 2 * key_pt.size )
				{
					is_valid = 0;
					break;
				}
			}
		}
#endif 
		if (is_valid == 0) continue;

		//计算掩码区域的均值和面积
		calc_roi_area_and_avg_by_mask(gray_img.data, matDst.data, &center.info.bd_rect, gray_img.cols, &region_area, &avg_val);
	
		//if (avg_val<detect_params->centre_gray_thr ) continue;

		//int search_rect_w = rect_roi.right - rect_roi.left;
		//int search_rect_h = rect_roi.bottom - rect_roi.top;

		center.info.area = region_area;
		key_pt.response = 0;

		if ( center.info.area>detect_params->max_area )	continue;
		if ( center.info.area<detect_params->min_area )	continue;

		if (rect_w >= 10 && rect_h >= 10)
		{
			if (center.info.area < 0.7* (rect_w*rect_h))  //填充率偏低
			{
				//continue;
			}
		}

		//长短轴比例
		//if ((center.info.major_axis / center.info.minor_axis) < 2) //主次轴比例
		{
			key_pt.response = 1;
		}

		key_pt.size = MAX(rect_w, rect_h) / 2;
		bak_pts.push_back(key_pt);

		//抠图处理
		float ratio = 0.25;
		int ext_sub_h = MIN( 15,MAX( 3,ratio*rect_h ) );
		int ext_sub_w = MIN( 15,MAX( 3,ratio*rect_w ) );
		rect_t rect_roi_sub;
	
		rect_roi_sub.left  = MAX(0,         center.info.bd_rect.left - ext_sub_w);
		rect_roi_sub.right = MIN(img_w - 1, center.info.bd_rect.left + rect_w + ext_sub_w);
		rect_roi_sub.top   = MAX(0,         center.info.bd_rect.top - ext_sub_h );
		rect_roi_sub.bottom= MIN(img_h - 1, center.info.bd_rect.top + rect_h + ext_sub_h );
		//roi_mat = cur_mask(Rect(rect_roi_sub.left, rect_roi_sub.top, rect_roi_sub.right - rect_roi_sub.left + 1, rect_roi_sub.bottom - rect_roi_sub.top + 1));
		//roi_img = gray_img(Rect(rect_roi_sub.left, rect_roi_sub.top, rect_roi_sub.right - rect_roi_sub.left + 1, rect_roi_sub.bottom - rect_roi_sub.top + 1));

		if (blob_info_vec && key_pt.response)
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
	}


	//重新
	//keypoints->clear();
	//for (iter = bak_pts.begin(); iter != bak_pts.end(); iter++)
	//{
	//	KeyPoint key_pt = *iter;
	//	keypoints->push_back(key_pt);
	//}

#ifdef _DEBUG_MODE
	IplImage* prgb_img = cvCreateImage(cvSize(img_w, img_h), 8, 3);
	float ratio = 0.7;
	for (i = 0; i < img_h; i++)
	{
		for (j = 0; j < img_w; j++)
		{
			cvSet2D(prgb_img, i, j, cvScalarAll(pimg[i*img_w + j]));
			if (matDst.data[i*img_w + j])
			{
				cvSet2D(prgb_img, i, j, cvScalar(255 * (1 - ratio) + ratio*pimg[i*img_w + j], ratio*pimg[i*img_w + j], ratio*pimg[i*img_w + j], 0));
			}
		}
	}

	cvNamedWindow("GROUP", 0);
	cvShowImage("GROUP", prgb_img);
	cvReleaseImage(&prgb_img);
#endif 


}



//
static  int create_detector(SimpleBlobDetector** blobDetector, blob_detect_params_t*detect_params )
{
	SimpleBlobDetector::Params params;

	params.minThreshold  = 16*5;
	params.maxThreshold  = 255;
	params.thresholdStep = 8;
	params.filterByArea  = true;
	params.minArea = detect_params->min_area;
	params.maxArea = detect_params->max_area;

	params.filterByColor = 1;
	params.blobColor = 255;

	params.filterByConvexity = 0;  //凸率
	params.filterByInertia = 0;	   //惯性比
	params.minInertiaRatio = 0.5;
	params.maxInertiaRatio = 2;
	params.filterByCircularity = 0;

	*blobDetector = new SimpleBlobDetector(params);
	(*blobDetector)->create("SimpleBlob");

	return 0;
}

//亮斑检测
int blob_detect_process(Mat img, blob_detect_params_t* params, vector<blob_info_t>* pret)
{
	Mat img_gray;
	Mat* pgray = NULL;
	//Mat output_img;
	vector<KeyPoint> keypoints;
	vector<KeyPoint>::iterator iter;
	vector<blob_info_t>::iterator iter2;
	SimpleBlobDetector * blobDetector= NULL;
	int i, j;

	//清空
	pret->clear();

	if (img.channels() != 1)
	{
		cv::cvtColor(img, img_gray, COLOR_RGB2GRAY);
		pgray = &img_gray;
	}
	else
	{
		pgray = &img;
	}

	//平滑处理
	if (params->blur_size > 0)
	{
		cv::GaussianBlur(img_gray, img_gray, cv::Size(params->blur_size, params->blur_size), 2);
	}

	//创建检测器
	create_detector(&blobDetector,params);

	//检测
	blobDetector->detect(img_gray, keypoints);

	//过滤处理
	//for (iter = keypoints.begin(); iter != keypoints.end(); iter++)
	//{
	//	iter->response = 1;
	//}
	calc_hat_feat(img_gray.data, &keypoints, img_gray.cols, img_gray.rows);
	region_group(img_gray, params, &keypoints, img_gray.cols, img_gray.rows, pret);

	/// 画图
#ifdef _DEBUG_MODE 
	Mat img_show;
	img.copyTo(img_show);

	for (iter = keypoints.begin(); iter != keypoints.end(); iter++)
	{
		//if (iter->response)
		{
			circle(img_show, iter->pt, iter->size, iter->response ? CV_RGB(0,0,255) : Scalar(0, 0, 0), iter->response ? 5 : 1, 8, 0);
		}
	}

	for (iter2 = pret->begin(); iter2 != pret->end();iter2++)
	{
		blob_info_t info = *iter2;
		cv::rectangle(img_show, cv::Point(info.bd_rect.left, info.bd_rect.top), cv::Point(info.bd_rect.right, info.bd_rect.bottom),CV_RGB(255,0,0),5,8,0);
	}


	namedWindow("BLOB_DETECT", 0);
	imshow("BLOB_DETECT", img_show);

#endif 

	delete blobDetector;

	return 0;
}
