#include "get_file_features.h"
#include "feature.h"
#include <stdio.h>
#include <stdlib.h>
#include <io.h>
#include <string.h>

#include "opencv2/core/core.hpp"  
#include "opencv2/imgproc/imgproc.hpp" 
#include "opencv2/highgui/highgui.hpp"

static void add_feat_to_file( FILE* fp,float* feat,int dim ,int label )
{
	int i;
	fprintf(fp,"%d ", label);

	for (i = 1; i <= dim;i++)
	{
		fprintf(fp, "%d:%.6f", i, feat[i-1]);
		if (i == dim)
		{
			fprintf(fp, "\n");
		}
		else
		{
			fprintf(fp, " ");
		}
	}

	fflush(fp);
}


static int generate_svm_train_dat_directory(char* path, FILE* fp, int label, int bin_nr, int feat_dim,int feat_type )
{
	FILE* fp_db = NULL;
	struct _finddata_t fd;
	char find_str[260];
	long handle;

	float* feat = new float[feat_dim];
	int counter = 0;

	sprintf(find_str, "%s\\*.jpg", path);

	handle = _findfirst(find_str, &fd);
	if (handle == -1) return 0;

	while (0 == _findnext(handle, &fd))
	{
		int k = 0;
		cv::Mat img_mat, img_gray,lbp_img,ang_img,mag_img;
		unsigned char* tmp_lbp_map = NULL;
		unsigned char* tmp_mag = NULL;
		unsigned char* tmp_ang = NULL;

		char full_fn[260];
		rect_t roi;

		if (strcmp(fd.name, "..") == 0 ||
			strcmp(fd.name, ".") == 0)
		{
			continue;
		}

		sprintf(full_fn, "%s/%s", path, fd.name);
		img_mat = cv::imread(full_fn);

		printf("scan file %s\n", full_fn);
		if (img_mat.empty())
		{
			printf("open pic failed!!! %s\n ", full_fn);
			continue;
		}

		cv::cvtColor( img_mat, img_gray, CV_RGB2GRAY);

		cv::GaussianBlur(img_gray, img_gray, cv::Size(3, 3), 2 );

		if (feat_type == 0)
		{
			lbp_img = cv::Mat::zeros(cv::Size(img_gray.cols, img_gray.rows), CV_8UC1);
			tmp_lbp_map = lbp_img.data;
			//计算lpb图像
			calc_lbp_map(img_gray.data, tmp_lbp_map, img_gray.rows, img_gray.cols);
		}
		if (feat_type == 1)
		{
			ang_img = cv::Mat::zeros(cv::Size(img_gray.cols, img_gray.rows), CV_8UC1);
			mag_img = cv::Mat::zeros(cv::Size(img_gray.cols, img_gray.rows), CV_8UC1);

			calc_hog_gradient_img(img_gray.data, mag_img.data, ang_img.data, img_gray.rows, img_gray.cols);
		}
		

		//计算特征
		roi.left = roi.top = 1;
		roi.right = img_gray.cols - 2;
		roi.bottom = img_gray.rows - 2;

		if (feat_type == 0)
		{
			calc_lbp_feat_roi(tmp_lbp_map, img_gray.cols, feat, bin_nr, &roi);
		}

		if (feat_type == 1)
		{
			calc_hog_feat_roi(mag_img.data, ang_img.data, img_gray.cols, feat, bin_nr, &roi);
		}

		//数据保存
		add_feat_to_file(fp, feat, feat_dim, label);

		cv::namedWindow("GRAY", 0);
		cv::imshow("GRAY", img_mat);

		if (!lbp_img.empty())
		{
			cv::namedWindow("LBP", 0);
			cv::imshow("LBP", lbp_img);
		}

		if (!mag_img.empty())
		{
			cv::namedWindow("GRAD_MAG", 0);
			cv::imshow("GRAD_MAG", mag_img);
		}


		cvWaitKey(1);
		counter++;

	}

	return counter;
}

int generate_svm_train_dat( char* simple_path, char* save_filename )
{
	int i;
	//遍历文件夹下所有的图片
	int sub_dir_label[] = { -1, 1 };
	char fiename[260];
	FILE*fp = fopen(save_filename, "wt");
	int total_sum = 0;

	int bin_nr = 8;
	int feat_dim = bin_nr * 10;
	int feat_type = 1;


	for (i = 0; i < sizeof(sub_dir_label) / sizeof(sub_dir_label[0]);i++)
	{
		sprintf(fiename, "%s\\%d", simple_path, sub_dir_label[i]);

		total_sum += generate_svm_train_dat_directory(fiename, fp, sub_dir_label[i], bin_nr, feat_dim, feat_type);

	}

	printf("total file =%d\n", total_sum);
	return 0;
}