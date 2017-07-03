#include "track.h"
#include<stdlib.h>
#include<vector>
#include "opencv2/video/tracking.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

#define MAX_POINT_NR (100)
#define MAX_TRACK_LEN (50)

typedef struct point_flt_tag
{
	float x;
	float y;
}point_flt_t;

typedef struct track_list_tag
{
	int first_update_idx;
	int last_update_idx;
	point_flt_t pts[MAX_TRACK_LEN];
}track_list_t;

typedef struct tracker_tag
{
	int frame_nr;
	Mat last_frame;
	vector<track_list_t*> track_list_vec;
}tracker_t;


int tracker_create(void** tracker)
{
	tracker_t* ptrack = new tracker_t;
	//ptrack->last_frame = Mat::zeros()
	*tracker = ptrack;

	ptrack->frame_nr = 0;

	return 0;
}

static void draw_track(tracker_t* ptrack, cv::Mat rgb_frame,int normal_w,int normal_h )
{
	int i, k;
	char title[16];
	vector<track_list_t*>::iterator list_iter;
	float plus_x = (float)rgb_frame.cols / normal_w;
	float plus_y = (float)rgb_frame.rows / normal_h;

	//if (ptrack->frame_nr<5) return;

	k = 0;
	for (list_iter = ptrack->track_list_vec.begin(); list_iter != ptrack->track_list_vec.end(); list_iter++,k++)
	{
		track_list_t* pt_list = *list_iter;
		point_flt_t cur_pt = pt_list->pts[ ptrack->frame_nr%MAX_TRACK_LEN ];

		if (pt_list->last_update_idx != ptrack->frame_nr ||
			pt_list->first_update_idx==ptrack->frame_nr ) continue;
		
		//轨迹
		
		//cv::polylines( )
#if 1
		for( i=ptrack->frame_nr;i> MAX( ptrack->frame_nr-MAX_TRACK_LEN+1,pt_list->first_update_idx+1);i-- )
		//for (i = ptrack->frame_nr; i> (ptrack->frame_nr-1); i--)
		{
			point_flt_t pt1 = pt_list->pts[i%MAX_TRACK_LEN];
			point_flt_t pt2 = pt_list->pts[(i-1)%MAX_TRACK_LEN];
			float dst = (pt1.x - pt2.x)*(pt1.x - pt2.x) +
				(pt1.y - pt2.y)*(pt1.y - pt2.y);
			if (dst > (80*80))
			{
				printf("error:%d !!! %d\n",k, i);
			}

			cv::line(rgb_frame, 
				Point(pt_list->pts[i%MAX_TRACK_LEN].x *plus_x,        pt_list->pts[i%MAX_TRACK_LEN].y*plus_y),
				Point(pt_list->pts[(i -1) % MAX_TRACK_LEN].x*plus_x, pt_list->pts[(i - 1) % MAX_TRACK_LEN].y*plus_y ),
				Scalar( 0, 0, 255, 0 ), 1, 8, 0);
		}
#endif 

		cv::circle(rgb_frame, Point(cur_pt.x*plus_x, cur_pt.y*plus_y), 3, Scalar(0, 0, 255), 1);

	}

	sprintf(title, "%d", ptrack->frame_nr);
	cv::putText(rgb_frame, title, cv::Point(15, 30), cv::FONT_HERSHEY_PLAIN, 2.0, Scalar(0, 255, 0, 0), 3);


	namedWindow("TRACK", 0);
	imshow("TRACK", rgb_frame);
}


int tracker_process(void* handle, cv::Mat rgb_frame )
{
	int i;
	int near_pt_nr = 0;
	tracker_t* ptrack = (tracker_t*)handle;

	cv::Mat cur_frame;
	vector<Point2f> prepoint, nextpoint, startpoint, startpoint_bak;
	vector<uchar> state;
	vector<float>err;
	vector<Point2f>::iterator iter, iter2;
	vector<track_list_t*>::iterator list_iter;

	int min_dist = 5;       //间距
	int lost_frame_nr = 5; //丢失帧数
	
	int pre_num = 0;
	
	//颜色转换
	cv::cvtColor(rgb_frame, cur_frame, cv::COLOR_RGB2GRAY);


	cv::resize(cur_frame, cur_frame, Size(704, 576));

	//平滑
	cv::GaussianBlur(cur_frame, cur_frame, cv::Size(3, 3), 2.5, 2.5);

	//第一帧不做处理
	if (ptrack->frame_nr==0 )// || ptrack->last_frame.empty())
	{
		
		goto FUNC_EXIT;
	}


	//获取当前的特征点
	//if (ptrack->frame_nr == 0)
	{
		goodFeaturesToTrack( ptrack->last_frame, prepoint, MAX_POINT_NR, 0.05, min_dist, Mat(), 3, false, 0.04 );
	}

	//特征点和历史点合并
	for (list_iter = ptrack->track_list_vec.begin(); list_iter != ptrack->track_list_vec.end(); list_iter++)
	{
		track_list_t* pt_list = *list_iter;
		point_flt_t pt = pt_list->pts[(ptrack->frame_nr-1) % MAX_TRACK_LEN];
		//继续跟踪的点
		startpoint.push_back( Point2f(pt.x, pt.y) );
	}

	pre_num = startpoint.size();
	startpoint_bak = startpoint;

	//当前图像的角点是否和历史点是否距离小于阈值,如果有新的点，则添加到历史点中
	for (iter = prepoint.begin(); iter != prepoint.end(); iter++)
	{
		int exist_min_dst = 0;
		for (iter2 = startpoint_bak.begin(); iter2 != startpoint_bak.end(); iter2++)
		{
			float dst = (iter->x - iter2->x) * (iter->x - iter2->x);
			dst += (iter->y - iter2->y) * (iter->y - iter2->y);
			if (dst <= (min_dist*min_dist))
			{
				exist_min_dst = 1;
				break;
			}
		}

		if (exist_min_dst == 0)
		{
			startpoint.push_back(*iter);
		}

		near_pt_nr += exist_min_dst;
	}

	printf("near pt =%d hist_pt_nr=%d cur_nr=%d after=%d\n", near_pt_nr, pre_num, prepoint.size(), startpoint.size());


	//更新
	//cornerSubPix(ptrack->last_frame, startpoint, Size( min_dist/2, min_dist/2 ), Size(-1, -1), TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03));

	//跟踪
	calcOpticalFlowPyrLK(ptrack->last_frame, cur_frame, startpoint, nextpoint, state, err, Size(21, 21), 3 );

	//跟踪轨迹更新
	for (i = 0; i < ptrack->track_list_vec.size(); i++)
	{
		//跟踪成功
		if (state[i] != 0)
		{
			ptrack->track_list_vec[i]->pts[ptrack->frame_nr%MAX_TRACK_LEN].x = nextpoint[i].x;
			ptrack->track_list_vec[i]->pts[ptrack->frame_nr%MAX_TRACK_LEN].y = nextpoint[i].y;
			ptrack->track_list_vec[i]->last_update_idx = ptrack->frame_nr;
		}
		else
		{
			if (ptrack->frame_nr == 0)
			{
				ptrack->track_list_vec[i]->pts[ptrack->frame_nr%MAX_TRACK_LEN].x = startpoint[i].x;
				ptrack->track_list_vec[i]->pts[ptrack->frame_nr%MAX_TRACK_LEN].y = startpoint[i].y;
			}
			else
			{
				ptrack->track_list_vec[i]->pts[ptrack->frame_nr%MAX_TRACK_LEN] = ptrack->track_list_vec[i]->pts[(ptrack->frame_nr - 1) % MAX_TRACK_LEN];
			}
		}
	}

	//新出现的目标
	for (; i < state.size(); i++)
	{
		if (state[i] != 0)
		{
			track_list_t* pnewlist = new track_list_t;
			pnewlist->pts[ptrack->frame_nr%MAX_TRACK_LEN].x = nextpoint[i].x;
			pnewlist->pts[ptrack->frame_nr%MAX_TRACK_LEN].y = nextpoint[i].y;
			pnewlist->first_update_idx = ptrack->frame_nr;
			pnewlist->last_update_idx = ptrack->frame_nr;

			if (ptrack->frame_nr == 1)
			{
				pnewlist->pts[0].x = nextpoint[i].x;
				pnewlist->pts[0].y = nextpoint[i].y;
			}

			ptrack->track_list_vec.push_back(pnewlist);
		}
		else
		{
			printf("firtst failed!!\n");
		}
	}

	//删除超时的目标
	printf("del %d\n", ptrack->track_list_vec.size());
	do{
		int del_nr = 0;
		for (list_iter = ptrack->track_list_vec.begin(); list_iter != ptrack->track_list_vec.end(); list_iter++)
		{
			track_list_t* pt_list = *list_iter;
			if ((ptrack->frame_nr - pt_list->last_update_idx) >= lost_frame_nr)
			{
				del_nr++;
				ptrack->track_list_vec.erase(list_iter);
				delete pt_list;
				break;
			}
		}
		if (del_nr == 0) break;
		else
		{
			//printf("del after %d\n", ptrack->track_list_vec.size());
		}
	} while(1);

	//printf(" update %d list_nr =%d \n", startpoint.size() - pre_num, ptrack->track_list_vec.size() );

	//
FUNC_EXIT:

	cur_frame.copyTo(ptrack->last_frame);
	draw_track(ptrack, rgb_frame, cur_frame.cols, cur_frame.rows);

	ptrack->frame_nr++;

	return 0;
}