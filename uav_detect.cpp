#include "uav_detect.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"


#include "type.h"

#define NORMAKL_IMG_W (704)
#define NORMAKL_IMG_H (576)
#define CENTER_WEIGHT (0.6)

using namespace cv;


typedef struct  uav_detect_params_tag
{
	int blur_size;             //图像平滑半径
	int min_area;              //最小面积
	int max_area;			   //最大面积
	int region_grow_thr;       //区域生长的阈值
	int centre_gray_thr;       //中心点亮度
}uav_detect_params_t;

typedef struct uav_obj_tag
{
	uav_rect_t rect;
	int area;
	int diff;  //和邻域的差值
	int centre_val; //中心点的值
	float score;
}uav_obj_t;


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



//云层区域过滤
static cv::Mat filter_cloud_uva2(cv::Mat pimg_gray)
{
	int i, j;
	int img_w = pimg_gray.cols;
	int img_h = pimg_gray.rows;
	int* proj_rows = new int[img_h];
	float* proj_rows_smooth = new float[img_h];
	int* pintegral = new int[img_h*img_w];

	int bd = 15;
	int calc_r = 5; //计算半径

	Mat img_filter = Mat::zeros(img_h, img_w, CV_8UC1);
	unsigned char* pimg_filter = (unsigned char*)img_filter.data;

	unsigned char* pimg = (unsigned char*)pimg_gray.data;

	float bd_plus = 2;
	calc_integral_img(pimg, pintegral, img_w, img_h);

	//计算行的均值
	for (i = bd; i < img_h - bd; i++)
	{
		proj_rows[i] = calc_block_sum(pintegral, bd, img_w - bd, i - calc_r, i + calc_r, img_w);
	}

	//数据补齐
	for (i = 0; i < bd; i++)
	{
		proj_rows[i] = proj_rows[bd];
		proj_rows[img_h - i - 1] = proj_rows[img_h - bd - 1];
	}

	//数据平滑
	for (i = 1; i < img_h - 1; i++)
	{
		proj_rows_smooth[i] = 0.5*proj_rows[i] + 0.25*proj_rows[i - 1] + 0.25*proj_rows[i + 1];
	}
	proj_rows_smooth[0] = proj_rows_smooth[1];
	proj_rows_smooth[img_h - 1] = proj_rows_smooth[img_h - 2];

	//二值化

	for (i = bd; i < img_h - bd; i++)
	{
		for (j = bd; j < img_w - bd; j++)
		{
			pimg_filter[i*img_w + j] = 255 * (pimg[i*img_w + j]>(MAX(20, proj_rows_smooth[i] * 0.2) + proj_rows_smooth[i]));
		}
	}

	//膨胀
	Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
	cv::dilate(img_filter, img_filter, kernel);
	cv::erode(img_filter, img_filter, kernel);

	//区域标记得到
#ifdef _DEBUG_MODE
	cv::namedWindow("FILTER", 0);
	cv::imshow("FILTER", img_filter);
#endif 

	delete[]proj_rows_smooth;
	delete[]proj_rows;
	delete[]pintegral;

	//返回前景的模板掩码图像
	return img_filter.clone();
}



//计算候选点附近的对比特性
static void calc_hat_feat(unsigned char* pimg, vector<KeyPoint>* keypoints, int img_w, int img_h)
{
	int k;
	vector<int> diff_vec;

	vector<KeyPoint>::iterator iter;
	int* pintegral = new int[img_h*img_w];
	float bd_plus = 1.5;
	int max_diff = 0;
	calc_integral_img(pimg, pintegral, img_w, img_h);

	for (iter = keypoints->begin(); iter != keypoints->end(); iter++)
	{
		int thr = 15;
		int x = iter->pt.x;
		int y = iter->pt.y;
		float size = MAX(5, iter->size);
		int left, top, right, bottom;
		int radius = MAX(3, MIN(15, size*bd_plus));
		int val[8];
		int grow_val_high = iter->angle;
		int grow_val_low =  iter->angle;

		grow_val_high = grow_val_high & 255;
		grow_val_low = (grow_val_low >> 8) & 255;


		//if (iter->response<1) continue;
		if (iter->response==0) continue;

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
		int diff_avg = 0;

		for (k = 0; k<8; k++)
		{
			big_nr += v11>(val[k] + thr);
			big_nr_low += v11 > (val[k] + thr_low);

			diff_avg += ( grow_val_high - val[k] );
		}

		diff_avg /= 8;

		int max_diff = 0;
		max_diff = MAX( max_diff, diff_avg);

		diff_vec.push_back(diff_avg);

#if 0
		if (diff_avg >= 90)
		{
			iter->response = 2;
		}
		else if (big_nr_low >= 8 && big_nr >= 1)
		{
			iter->response = 1;
		}
#endif 

	}

	delete[] pintegral;
}


//区域生长,返回生长掩码图和区域的均值和面积
static  Mat seed_region_group_uav(unsigned char* pimg, KeyPoint* key_pt, rect_t* rect, int img_w, int img_h, int th, int *region_area, int *region_avg_val)
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

	th = MAX(th, nSrcValue*0.25);

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


//查找极值点,在key_pt附近半径内 ratio*img_h(img_w)   
static int find_near_peak(unsigned char* pimg, int img_w, int img_h, KeyPoint*key_pt, int bd_width, float ratio)
{
	int i, j;
	rect_t search_roi;
	int inner_radius = key_pt->size;

	int extern_radius= 3*inner_radius;

	int ref_val = ((int)key_pt->angle)&255;
	int ref_small = MAX(50,ref_val - ref_val*0.2);
	int nr = 0;

	search_roi.left = key_pt->pt.x - extern_radius;// img_w*ratio;
	search_roi.right = key_pt->pt.x + extern_radius;// img_w*ratio;
	search_roi.top = key_pt->pt.y - extern_radius;// img_h*ratio;
	search_roi.bottom = key_pt->pt.y + extern_radius;// img_h*ratio;

	search_roi.left  = MAX(bd_width, search_roi.left);
	search_roi.right = MIN(img_w-bd_width, search_roi.right);
	search_roi.top   = MAX(bd_width, search_roi.top);
	search_roi.bottom= MIN(img_h-bd_width, search_roi.bottom);

	for (i = search_roi.top; i <= search_roi.bottom;i++)
	{
		for (j = search_roi.left; j <= search_roi.right; j++)
		{
			if (ABS(i - key_pt->pt.y) <= inner_radius ||
				ABS(j - key_pt->pt.x) <= inner_radius)
			{
				continue;
			}

			nr += (pimg[i*img_w + j] > ref_small);
		}
	}

	return nr;
}

//计算候选点附近的对比特性
static void region_group_uav(Mat gray_img, uav_detect_params_t*detect_params, vector<KeyPoint>* keypoints, vector<uav_obj_t>* ret_pts)
{
	int i, j;
	vector<KeyPoint>::iterator iter;
	vector<uav_obj_t>::iterator iter2;
	vector<KeyPoint> bak_pts;
	
	int img_w = gray_img.cols;
	int img_h = gray_img.rows;

	unsigned char* pimg = gray_img.data;

	Mat matDst = Mat::zeros(cv::Size(img_w, img_h), CV_8UC1);	//创建一个空白区域，填充为黑色
	int thr_small = detect_params->region_grow_thr;
	int thr_big = detect_params->region_grow_thr;
	char filename[260];
	
	float area_thr_ratio = 3.0f;


	//cv::Mat filter_mask = filter_cloud_uva2( gray_img );

	//遍历所有的候选点
	for (iter = keypoints->begin(); iter != keypoints->end(); iter++ )
	{
		KeyPoint key_pt = *iter;
		rect_t rect_roi, rect_roi_sub;
		Mat cur_mask;

		int k = 5;
		int area_small, avg_val_small;
		int area_big, avg_val_big;

		int pos_y = key_pt.pt.y;
		int pos_x = key_pt.pt.x;

		int seed_v = pimg[pos_y*img_w+pos_x];
		int roi_area;
		//如果无效的点不考虑
		//if (key_pt.response == 0) continue;
		//key_pt.response = 0;

		rect_roi.left = MAX(0, key_pt.pt.x - k * key_pt.size);
		rect_roi.right = MIN(img_w - 1, key_pt.pt.x + k * key_pt.size);
		rect_roi.top = MAX(0, key_pt.pt.y - k * key_pt.size);
		rect_roi.bottom = MIN(img_h - 1, key_pt.pt.y + k * key_pt.size);

		roi_area = (rect_roi.bottom - rect_roi.top)*(rect_roi.right - rect_roi.left);

		//区域生长
		thr_small = MAX(10, seed_v*0.1);

		cur_mask = seed_region_group_uav(pimg, &key_pt, &rect_roi, img_w, img_h, thr_small, &area_small, &avg_val_small);
		

		thr_big = MAX(2*thr_small,seed_v*0.3);
		cur_mask = seed_region_group_uav(pimg, &key_pt, &rect_roi, img_w, img_h, thr_big, &area_big, &avg_val_big);

		key_pt.angle = avg_val_small + avg_val_big*256;

		key_pt.response = ((area_big < area_thr_ratio * area_small) && (4 * area_big<roi_area) && area_big<detect_params->max_area);
		key_pt.class_id = area_small;
		key_pt.octave = area_big;

		bak_pts.push_back(key_pt);

		//得到局部小图,然后对小图进行findcounter操作
		//Mat roi_mat = cur_mask(Rect(rect_roi.left, rect_roi.top, rect_roi.right - rect_roi.left, rect_roi.bottom - rect_roi.top));
		//最终以大图的为准
		matDst |= cur_mask;
	}

	//对生长后的掩码处理
	Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
	
	//膨胀
	cv::dilate(matDst, matDst, kernel);
	cv::erode(matDst, matDst, kernel);

	//matDst |= filter_mask;

	vector < vector<Point> > contours;
	cv::findContours( matDst, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
	
	
	//calc_hat_feat(pimg,&bak_pts,img_w,img_h );

#ifdef _DEBUG_MODE
	{
		for (size_t contourIdx = 0; contourIdx < contours.size(); contourIdx++)
		{
			cv::drawContours(matDst, contours, contourIdx, Scalar((contourIdx+1)*255 / contours.size(), 0, 0, 0), 3);
		}
	}
#endif 

	//
	for (iter = bak_pts.begin(); iter != bak_pts.end(); iter++)
	{
		KeyPoint key_pt = *iter;
		int near_pk_nr = 0;
		uav_obj_t valid_obj;
		if (key_pt.response == 0) goto APPEND;

		//near_pk_nr = find_near_peak(pimg, img_w, img_h, &key_pt, 10, 1.0 / 8.0f);
		//if (near_pk_nr <= 4)
		//{
		//	key_pt.response = 1;
		//	goto APPEND;
		//}

		//key_pt.response = 1;

		for (size_t contourIdx = 0; contourIdx < contours.size(); contourIdx++)
		{
			if (cv::pointPolygonTest(contours[contourIdx], cv::Point2f(key_pt.pt.x, key_pt.pt.y), false)>0)
			{
				
				//Moments moms = moments(Mat(contours[contourIdx]));
				double contour_area = cv::contourArea( Mat(contours[contourIdx]) );
				key_pt.response = (1.5*area_thr_ratio*key_pt.class_id > contour_area && contour_area<detect_params->max_area);

				if (key_pt.response)
				{
					vector<Point>::iterator pt_iter;
					int left, top, right, bottom;
					left = right = contours[contourIdx][0].x;
					top = bottom = contours[contourIdx][0].y;
					for (pt_iter = contours[contourIdx].begin(); pt_iter != contours[contourIdx].end(); pt_iter++)
					{
						left  = MIN(left,pt_iter->x);
						right = MAX(right,pt_iter->x);
						top   = MIN(top, pt_iter->y);
						bottom= MAX(bottom, pt_iter->y);
					}

					valid_obj.area = contour_area;
					valid_obj.rect.left = left;
					valid_obj.rect.right = right;
					valid_obj.rect.top = top;
					valid_obj.rect.bottom = bottom;
				}

				//key_pt.response = sqrt(moms.m00);
				break;
			}
		}

	APPEND:
		if (key_pt.response)
		{
			ret_pts->push_back(valid_obj);
		}

#ifdef _DEBUG_MODE
		cv::circle(matDst, cv::Point(key_pt.pt.x, key_pt.pt.y), 3, Scalar(128, 128, 128), 1);
#endif 

	}


	//计算差异
	int* pintegral = new int[img_h*img_w];
	
	calc_integral_img(pimg, pintegral, img_w, img_h);

	for (iter2 = ret_pts->begin(); iter2 != ret_pts->end(); iter2++)
	{
		int left, top, right, bottom;
		
		int k, sub_w,sub_h;
		int val[8];

		left = iter2->rect.left;
		right = iter2->rect.right;
		top = iter2->rect.top;
		bottom = iter2->rect.bottom;

		sub_w = right - left;
		sub_h = bottom - top;

		val[0] = calc_block_sum( pintegral, MAX(0,left - sub_w), left,          MAX( 0, top - sub_h ), top, img_w);
		val[1] = calc_block_sum( pintegral, left,                right,         MAX( 0, top - sub_h ), top, img_w);
		val[2] = calc_block_sum( pintegral, right, MIN( img_w-1,right + sub_w), MAX( 0, top - sub_h ), top, img_w);

		val[3] = calc_block_sum( pintegral, MAX(0, left - sub_w), left,         top, bottom, img_w);
		int v11 = calc_block_sum(pintegral, left,                 right,        top, bottom, img_w);
		val[4] = calc_block_sum( pintegral, right, MIN(img_w - 1,right + sub_w),top, bottom, img_w);

		val[5] = calc_block_sum( pintegral, MAX(0, left - sub_w), left,          bottom, MIN( img_h-1, bottom + sub_h ), img_w );
		val[6] = calc_block_sum( pintegral, left, right,						 bottom, MIN( img_h-1, bottom + sub_h ), img_w );
		val[7] = calc_block_sum( pintegral, right, MIN(img_w - 1, right + sub_w),bottom, MIN( img_h-1, bottom + sub_h ), img_w );

		//比较阈值
		int diff_avg = 0;

		for (k = 0; k<8; k++)
		{
			
			diff_avg += (v11 - val[k]);
		}

		diff_avg /= 8;
		iter2->diff = diff_avg;
		iter2->centre_val = v11;
		iter2->score = v11 / 255.0*CENTER_WEIGHT + (1.0 - CENTER_WEIGHT)*diff_avg / 255.0;

		if (diff_avg < 15)
		{
			iter2->score = 0;
		}
	}


#ifdef _DEBUG_MODE
	cv::namedWindow("MASK_GROW", 0);
	cv::imshow("MASK_GROW", matDst);
#endif 

	delete[]pintegral;

}

//创建检测器
static  int create_detector(SimpleBlobDetector** blobDetector, uav_detect_params_t*detect_params)
{
	SimpleBlobDetector::Params params;

	params.minThreshold = 16 * 5;
	params.maxThreshold = 255;
	params.thresholdStep = 16;
	params.filterByArea = true;
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


//云层区域过滤
static void filter_cloud_uva(cv::Mat frame_i)
{
	Mat img_gray;
	Mat blur_big;
	Mat blur_small;

	float sigma_small = 2.5;
	float sigma_big = sigma_small * 5;

	int i,j;
	int img_w = frame_i.cols;
	int img_h = frame_i.rows;
	Mat img_diff = cv::Mat(img_h, img_w, CV_8UC1);

	unsigned char* ptmp_diff = img_diff.data;
	unsigned char* psmall;
	unsigned char* pbig;
	unsigned char* psrc;
	double avg, var;
	int area;

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

	filter_cloud_uva2(img_gray);

#if 0
	for (i = 0; i<img_h*img_w; i++)
	{
		//ptmp_diff[i] = (psmall[i] >(pbig[i] + 10))*((unsigned char*)img_gray.data)[i];// -pbig[i]);
		ptmp_diff[i] = (psmall[i] >(pbig[i] + 10));
	}


	/*************/
	avg = 0;
	var = 0;
	area = 0;
	for (i=sigma_big;i<img_h-sigma_big;i++)
	{
		for (j=sigma_big;j<img_w-sigma_big;j++)
		{
			if (ptmp_diff[i*img_w + j])
			{
				avg += img_gray.data[i*img_w + j];
				area++;
			}
		}
	}

	avg /= (area+1);

	for (i=sigma_big;i<img_h-sigma_big;i++)
	{
		for (j=sigma_big;j<img_w-sigma_big;j++)
		{
			if (ptmp_diff[i*img_w + j])
			{
				var += (avg - img_gray.data[i*img_w + j])*(avg - img_gray.data[i*img_w + j]);
			}
		}
	}
	var = sqrt(var/(area+1));

	printf("%3f,%3f %d\n", avg, var,area);

	//cv::watershed( frame_i, img_diff);
#if 0
	//对生长后的掩码处理
	Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));

	//膨胀
	cv::dilate(img_diff, img_diff, kernel);
	cv::erode(img_diff, img_diff, kernel);
	vector < vector<Point> > contours;
	cv::findContours(img_diff, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

	for (size_t contourIdx = 0; contourIdx < contours.size(); contourIdx++)
	{
		cv::drawContours(img_diff, contours, contourIdx, Scalar((contourIdx + 1) * 255 / contours.size(), 0, 0, 0), 3);
	}
#endif 


	cv::imshow("blur_small", blur_small);
	cv::imshow("blur_big", blur_big);
	cv::imshow("img_diff",255* img_diff);
#endif 

}

//
static int compare( const void* n1, const void* n2)
{
	uav_rect_t* p1 = (uav_rect_t*)n1;
	uav_rect_t* p2 = (uav_rect_t*)n2;
	return p1->score < p2->score;
}


int uav_detect_process(cv::Mat img, vec_rect_t* pret_vec)
{
	cv::Mat normal_img;
	Mat img_gray,proc_img;
	Mat morp_ele;
	SimpleBlobDetector * blobDetector = NULL;
	uav_detect_params_t detect_params;
	vector<KeyPoint> keypoints;
	vector<uav_obj_t> keypoints_ret;
	vector<KeyPoint>::iterator iter0;
	vector<uav_obj_t>::iterator iter;

	memset(&detect_params, 0, sizeof(detect_params));

	detect_params.blur_size = 3;
	detect_params.centre_gray_thr = 100;
	detect_params.min_area = 3 * 3;
	detect_params.max_area = 30 * 30;
	detect_params.region_grow_thr = 30;

	//创建检测器
	create_detector(&blobDetector, &detect_params);

	//清空
	pret_vec->clear();

	//图像大小归一化
	cv::resize( img, normal_img, cv::Size(NORMAKL_IMG_W, NORMAKL_IMG_H));

	//转化为灰度图像
	if (normal_img.channels() != 1)
	{
		cv::cvtColor(normal_img, img_gray, COLOR_RGB2GRAY);
	}
	else
	{
		img_gray = normal_img;
	}

	cv::GaussianBlur(img_gray, img_gray, cvSize(detect_params.blur_size, detect_params.blur_size), 2.5, 2.5);
	//filter_cloud_uva(img_gray);

	//morp_ele = cv::getStructuringElement(cv::MORPH_RECT, Size(15,15)); // cvSize(sqrt(detect_params.max_area), sqrt(detect_params.max_area)));
	//cv::morphologyEx(img_gray, proc_img, cv::MORPH_TOPHAT, morp_ele);
	//cv::namedWindow("TOPHAT", 0);
	//cv::imshow("TOPHAT", proc_img);

	//blobDetector->detect(proc_img, keypoints);
	blobDetector->detect(img_gray, keypoints);

	//图像生长处理
	region_group_uav(img_gray, &detect_params, &keypoints, &keypoints_ret);
	//calc_hat_feat( img_gray.data, &keypoints, img_gray.cols, img_gray.rows );

	uav_rect_t* ptmp = (uav_rect_t*)malloc( sizeof(*ptmp)*keypoints_ret.size() );
	int k = 0;
	for (iter = keypoints_ret.begin(); iter != keypoints_ret.end(); iter++)
	{
		uav_obj_t kpt = *iter;
		
		ptmp[k].left = (kpt.rect.left)*img.cols / img_gray.cols;
		ptmp[k].right = (kpt.rect.right)*img.cols / img_gray.cols;
		ptmp[k].top = (kpt.rect.top)*img.rows / img_gray.rows;
		ptmp[k].bottom = (kpt.rect.bottom)*img.rows / img_gray.rows;
		ptmp[k].score = kpt.score * 10000;
		k++;
	}

	//排序
	qsort(ptmp, k, sizeof(*ptmp), compare);

	for (k = 0; k < keypoints_ret.size();k++)
	{
		int bd_thr = 5;
		uav_rect_t ret_rect = ptmp[k];
		if (ret_rect.left<bd_thr || ret_rect.right>img.cols - bd_thr ||
			ret_rect.top<bd_thr || ret_rect.bottom>img.rows - bd_thr)
			continue;

		if (ret_rect.score>0.1)
		{
			pret_vec->push_back(ret_rect);
		}
	}

	free(ptmp);

	

#ifdef _DEBUG_MODE
	{

		for (iter0 = keypoints.begin(); iter0 != keypoints.end(); iter0++)
		{
			cv::circle(normal_img, cv::Point(iter0->pt.x, iter0->pt.y), 3, Scalar(0, 255, 0), 3);
		}

		for (iter = keypoints_ret.begin(); iter != keypoints_ret.end(); iter++)
		{
			char str[16];
			cv::rectangle(normal_img, cv::Point(iter->rect.left, iter->rect.top), cv::Point(iter->rect.right, iter->rect.bottom), Scalar(0, 0, 255), 3);

			sprintf(str, "diff:%d", iter->diff);
			cv::putText(normal_img, str, Point( iter->rect.right+5, iter->rect.bottom ), cv::FONT_HERSHEY_PLAIN, 1.0, Scalar(0, 0, 255), 2, 8);

			sprintf(str, "avg:%d", iter->centre_val);
			cv::putText(normal_img, str, Point(iter->rect.right + 5, iter->rect.bottom+20), cv::FONT_HERSHEY_PLAIN, 1.0, Scalar(0, 0, 255), 2, 8);


		}

		cv::namedWindow("BLOB_DETECTOR", 0);
		cv::imshow("BLOB_DETECTOR", normal_img);

		//cv::namedWindow("TOPHAT", 0);
		//cv::imshow("TOPHAT", proc_img);
	}
#endif 




	return pret_vec->size();
}
