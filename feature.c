#include "feature.h"
#include <string.h>
#include <math.h>
#include <stdio.h>

#define MAG_THR (0.5f)

//映射表
static const int lbp_table[256] = 
{ 
	1, 2, 3, 4, 5, 0, 6, 7, 8, 0, 0, 0, 9, 0, 10, 11, 12, 0, 0, 0, 0, 0, 0, 0, 13, 0, 0, 0, 14, 0, 15, 16, 17, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 0, 0, 0, 0, 0, 0, 0, 19, 0, 0, 0, 20, 0, 21, 22, 23, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25,
	0, 0, 0, 0, 0, 0, 0, 26, 0, 0, 0, 27, 0, 28, 29, 30, 31, 0, 32, 0, 0, 0, 33, 0, 0, 0, 0, 0, 0, 0, 34, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 35, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 36, 37, 38, 0, 39, 0, 0, 0, 40, 0, 0, 0, 0, 0, 0, 0, 41, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 42,
	43, 44, 0, 45, 0, 0, 0, 46, 0, 0, 0, 0, 0, 0, 0, 47, 48, 49, 0, 50, 0, 0, 0, 51, 52, 53, 0, 54, 55, 56, 57, 58
};


//归一化
static int normal_descrp( float *phist,int bin_nr )
{
	int i;
	float sum = 1e-6;

	for (i = 0; i < bin_nr;i++)
	{
		sum += phist[i] * phist[i];
	}

	sum = 1.0f / sqrt(sum);
	for (i = 0; i < bin_nr; i++)
	{
		phist[i] *= sum;
	}

	return 0;
}

//计算图像的lbp值
int calc_lbp_map( unsigned char* pimg, unsigned char* plbp_map, int img_h, int img_w)
{
	int i, j;
	int thr = 3;
	unsigned char* pcur, *pnext, *plast;
	memset(plbp_map, 0, sizeof(*plbp_map)*img_h*img_w);


	plast = pimg;
	pcur = plast + img_w;
	pnext = pcur + img_w;
	plbp_map += img_w;

	for (i = 1; i < img_h - 1; i++)
	{
		for (j = 1; j < img_w - 1; j++)
		{
			int lbp_v = 0;
			int ref_v = (pcur[j] + thr);
			lbp_v |= 1 * (plast[j - 1] > ref_v);
			lbp_v |= 2 * (plast[j]     > ref_v);
			lbp_v |= 4 * (plast[j + 1] > ref_v);
			lbp_v |= 8 * (pcur[j - 1] > ref_v);
			lbp_v |= 16 * (pcur[j + 1] > ref_v);
			lbp_v |= 32 * (pnext[j - 1] > ref_v);
			lbp_v |= 64 * (pnext[j] > ref_v);
			lbp_v |= 128 * (pnext[j + 1] > ref_v);

			plbp_map[j] = lbp_table[lbp_v];
		}

		pcur += img_w;
		plast += img_w;
		pnext += img_w;
		plbp_map += img_w;
	}

	return 0;
}

//获取图像roi区域内的lbp直方图
// proi_rect为全闭区间
static int calc_lbp_hist_roi(unsigned char* plbp_map, float *phist, int img_w, int bin_nr, rect_t* proi_rect)
{
	int i, j;
	float sum_val = 0;
	memset( phist, 0, sizeof(*phist)*bin_nr );
	plbp_map += proi_rect->top*img_w;

	for ( i = proi_rect->top; i <= proi_rect->bottom;i++ )
	{
		for (j = proi_rect->left; j <= proi_rect->right; j++)
		{
			phist[plbp_map[j]]++;
		}
	}

	//归一化处理
	for (i = 0; i < bin_nr;i++)
	{
		sum_val += phist[i];
	}

	for (i = 0; i < bin_nr; i++)
	{
		phist[i] /= sum_val;
	}

	return bin_nr;
}


//获取图像roi区域内的lbp直方图
// proi_rect为全闭区间
int calc_lbp_feat_roi( unsigned char* plbp_map, int img_w, float *phist, int bin_nr, rect_t* proi_rect )
{
	//图像分块
	//分解为3*3的

	int total_feat_dim = 0;
	int i, j;
	int block_h = proi_rect->bottom - proi_rect->top + 1;
	int block_w = proi_rect->right - proi_rect->left + 1;

	int x_pos[3] = { proi_rect->left, (proi_rect->left + proi_rect->right) / 2 - block_w / 4, proi_rect->right - (block_w)/2 + 1 };
	int y_pos[3] = { proi_rect->top,  (proi_rect->top + proi_rect->bottom) / 2 - block_h / 4, proi_rect->bottom - (block_h)/2 + 1 };
	float* tmp_hist = phist;

	for (i = 0; i < 3;i++)
	{
		for (j = 0; j < 3;j++)
		{
			rect_t sub_rect_roi;
			sub_rect_roi.top  = y_pos[i];
			sub_rect_roi.left = x_pos[j];

			sub_rect_roi.bottom = sub_rect_roi.top  + (block_h / 2)-1;
			sub_rect_roi.right  = sub_rect_roi.left + (block_w / 2) - 1;

			calc_lbp_hist_roi( plbp_map, tmp_hist, img_w, bin_nr, &sub_rect_roi);
			tmp_hist += bin_nr;
		}
	}

	calc_lbp_hist_roi(plbp_map, tmp_hist, img_w, bin_nr, proi_rect);

	return bin_nr * 10;
}


int calc_hog_gradient_img(unsigned char* pimg, unsigned char* pgrad_mag, unsigned char* pgrad_ang, int img_h, int img_w)
{
	int i, j;
	unsigned char* plast = pimg;
	unsigned char* pcur = plast + img_w;
	unsigned char* pnext = pcur + img_w;
	int mag_thr = 5;

	unsigned char* pgrad_mag_ptr = pgrad_mag + img_w;
	unsigned char* pgrad_ang_ptr = pgrad_ang + img_w;

	memset( pgrad_mag, 0, sizeof(*pgrad_mag)*img_h*img_w );
	memset( pgrad_ang, 0, sizeof(*pgrad_ang)*img_h*img_w );

	for (i = 1; i < img_h - 1;i++)
	{
		for (j = 1; j < img_w - 1;j++)
		{
			int dx = pcur[j + 1] - pcur[j - 1];
			int dy = pnext[j] - plast[j];
			int angle = 0;
			int grad_m = MAX(ABS(dx), ABS(dy));
			
			if ( grad_m>mag_thr )
			{
				if (dx==0)
				{
					if (dy > 0)
						angle = 90;
					else
						angle = 270;
				}
				else
				{
					double angle_fl = atan( (float)dy / dx);
					if (dy > 0)
					{
						angle = angle_fl * 180 / 3.1415926;
					}
					else
					{
						angle = angle_fl * 180 / 3.1415926 + 180;
					}
				}

				while (angle>=360)
				{
					angle -= 360;
				};

				while (angle < 0)
				{
					angle += 360;
				};

				pgrad_mag_ptr[j] = grad_m;
				pgrad_ang_ptr[j] = angle / 45;
				if (pgrad_ang_ptr[j] >= 8 || pgrad_ang_ptr[j] < 0)
				{
					printf("!!!!\n");
				}
			}
		}

		pgrad_mag_ptr += img_w;
		pgrad_ang_ptr += img_w;
		pcur += img_w;
		pnext += img_w;
		plast += img_w;
	}

	return 0;

}


//获取图像roi区域内的lbp直方图
// proi_rect为全闭区间
static int calc_hog_hist_roi(unsigned char* pgrad_mag, unsigned char* pgrad_ang, float *phist, int img_w, int bin_nr, rect_t* proi_rect)
{
	int i, j;
	float sum_val = 0;
	float sum_val2 = 0;
	memset(phist, 0, sizeof(*phist)*bin_nr);
	pgrad_ang += proi_rect->top*img_w;
	pgrad_mag += proi_rect->top*img_w;

	for (i = proi_rect->top; i <= proi_rect->bottom; i++)
	{
		for (j = proi_rect->left; j <= proi_rect->right; j++)
		{
			phist[pgrad_ang[j]] += pgrad_mag[j];
		}
	}

	//归一化处理
	//normal_descrp( phist, bin_nr );

	//for (i = 0; i < bin_nr; i++)
	//{
	//	phist[i] = MIN( phist[i], MAG_THR );
	//}
	//normal_descrp(phist, bin_nr);

	return bin_nr;
}



int calc_hog_feat_roi(unsigned char* pgrad_mag, unsigned char* pgrad_ang,int img_w, float *phist, int bin_nr, rect_t* proi_rect)
{
	int total_feat_dim = 0;
	int i, j;
	int block_h = proi_rect->bottom - proi_rect->top + 1;
	int block_w = proi_rect->right - proi_rect->left + 1;

	int x_pos[3] = { proi_rect->left, (proi_rect->left + proi_rect->right) / 2 - block_w / 4, proi_rect->right - (block_w) / 2 + 1 };
	int y_pos[3] = { proi_rect->top, (proi_rect->top + proi_rect->bottom) / 2 - block_h / 4, proi_rect->bottom - (block_h) / 2 + 1 };
	float* tmp_hist = phist;

	for (i = 0; i < 3; i++)
	{
		for (j = 0; j < 3; j++)
		{
			rect_t sub_rect_roi;
			sub_rect_roi.top = y_pos[i];
			sub_rect_roi.left = x_pos[j];

			sub_rect_roi.bottom = sub_rect_roi.top + (block_h / 2) - 1;
			sub_rect_roi.right = sub_rect_roi.left + (block_w / 2) - 1;

			calc_hog_hist_roi(pgrad_mag, pgrad_ang, tmp_hist, img_w, bin_nr, &sub_rect_roi);
			tmp_hist += bin_nr;
		}
	}

	calc_hog_hist_roi(pgrad_mag, pgrad_ang, tmp_hist, img_w, bin_nr, proi_rect);

	//归一化处理
	normal_descrp(phist, bin_nr*10);

	for (i = 0; i < 10*bin_nr; i++)
	{
		phist[i] = MIN( phist[i], MAG_THR );
	}
	normal_descrp(phist, 10*bin_nr);


	return bin_nr * 10;
}