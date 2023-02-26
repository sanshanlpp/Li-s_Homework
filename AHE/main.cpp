#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

void My_equalizeHist(Mat& src, Mat& dst);
void show_histogram(Mat& img, Mat& Hist);
void My_LocalHE(Mat& src, Mat& dst, const int m);
void My_AHE(Mat& src, Mat& dst, const int m);
void My_CHE(Mat& src, Mat& dst, const float clip_limit);
void My_CLAHE(Mat& src, Mat& dst, const float clip_limit, const int m);

int main()
{
    Mat src = imread("E://vis_38.png",0);
    if (src.empty()){
        return -1;
    }

    Mat dst1, dst2, dst3, dst4, dst5, Hist, Hist1, Hist2, Hist3, Hist4, Hist5;
    dst1 = Mat::zeros(src.size(), src.type());   //新建一个空矩阵（全为0）
    dst2 = Mat::zeros(src.size(), src.type());   //新建一个空矩阵（全为0）
    dst3 = Mat::zeros(src.size(), src.type());
    dst4 = Mat::zeros(src.size(), src.type());
    dst5 = Mat::zeros(src.size(), src.type());

    show_histogram(src, Hist);


    double t1 = (double)getTickCount();  //start timing

    My_equalizeHist(src, dst1);

    t1 = (double)getTickCount() - t1;
    double time1 = (t1 * 1000) / ((double)getTickFrequency());
    cout << "The time of HE is " << time1 << "ms." << endl; //finish timing and show

    //---------------------------------------------
    double t2 = (double)getTickCount();  //start timing

    My_LocalHE(src, dst2, 32);

    t2 = (double)getTickCount() - t2;
    double time2 = (t2 * 1000) / ((double)getTickFrequency());
    cout << "The time of LocalHE is " << time2 << "ms." << endl; //finish timing and show
    //---------------------------------------------
    double t3 = (double)getTickCount();  //start timing

    My_AHE(src, dst3, 16);

    t3 = (double)getTickCount() - t3;
    double time3 = (t3 * 1000) / ((double)getTickFrequency());
    cout << "The time of AHE is " << time3 << "ms." << endl; //finish timing and show

    //---------------------------------------------
    double t4 = (double)getTickCount();  //start timing

    My_CHE(src, dst4, 2.0);

    t4 = (double)getTickCount() - t4;
    double time4 = (t4 * 1000) / ((double)getTickFrequency());
    cout << "The time of CHE is " << time4 << "ms." << endl; //finish timing and show
    //---------------------------------------------
    double t5 = (double)getTickCount();  //start timing

    My_CLAHE(src, dst5, 4.0, 32);

    t5 = (double)getTickCount() - t5;
    double time5 = (t5 * 1000) / ((double)getTickFrequency());
    cout << "The time of CLAHE is " << time5 << "ms." << endl; //finish timing and show

    show_histogram(dst1, Hist1);
    show_histogram(dst2, Hist2);
    show_histogram(dst3, Hist3);
    show_histogram(dst4, Hist4);
    show_histogram(dst5, Hist5);
    //---------------------------------------------
    //display the achievements
    imshow("src",src);
    imshow("HE",dst1);
    imshow("localHE",dst2);
    imshow("AHE",dst3);
    imshow("CHE",dst4);
    imshow("CLAHE",dst5);

    imshow("raw_Hist",Hist);
    imshow("HEHist",Hist1);
    imshow("localHEHist",Hist2);
    imshow("AHEHist",Hist3);
    imshow("CHEHist",Hist4);
    imshow("CLAHEHist",Hist5);

//    imwrite("E://dst1.png",dst1);
//    imwrite("src.jpg",src);
//    imwrite("HE.jpg",dst1);
//    imwrite("localHE.jpg",dst2);
//    imwrite("AHE.jpg",dst3);
//    imwrite("CHE.jpg",dst4);
//    imwrite("CLAHE.jpg",dst5);

//    imwrite("raw_Hist.jpg",Hist);
//    imwrite("HEHist.jpg",Hist1);
//    imwrite("localHEHist.jpg",Hist2);
//    imwrite("AHEHist.jpg",Hist3);
//    imwrite("CHEHist.jpg",Hist4);
//    imwrite("CLAHEHist.jpg",Hist5);
    waitKey(0);

    return 0;
}

void My_equalizeHist(Mat& src, Mat& dst)
{
    int h = src.rows;
    int w = src.cols;
    int size = h * w;
//***********step1:像素灰度级数统计************
    int NumPixel[256] = {0};
    for(int i = 0; i < h; i++){
        for(int j = 0; j < w; j++){
            int temp = src.at<uchar>(i, j);
            NumPixel[temp] = NumPixel[temp] + 1;
        }
    }
//**********step2:进行像素灰度级数累计统计***********
    int CumPixel[256] = {0};
    for(int i = 0; i < 256; i++){
        if(i == 0)
            CumPixel[i] = NumPixel[i];
        else
            CumPixel[i] = CumPixel[i-1] + NumPixel[i];
    }
//************step3:对灰度值进行映射***********
/*    dst = Mat::zeros(src.size(), src.type()); */  //新建一个空矩阵（全为0）
    for(int i = 0; i < h; i++){
        for(int j = 0; j < w; j++){
            int temp = src.at<uchar>(i, j);
            int a = CumPixel[temp];
            double b = a *1.0 / size;
            int result = b * 255;
            dst.at<uchar>(i, j) = result;
        }
    }

}

void My_LocalHE(Mat& src, Mat& dst, const int m)
{
    Mat im_pad = src.clone();;  //copy
//    const int h = src.rows, w = src.cols;
//    const int rdh = h % m, rdw = w % m;
//    const int pdh = rdh == 0 ? h : h + m - rdh;
//    const int pdw = rdw == 0 ? w : w + m - rdw;
//    const int res_h = pdh - h, res_w = pdw - w;
//    const int top = res_h / 2, left = res_w / 2;
//    const int bottom = res_h - top, right = res_w - left;
//    copyMakeBorder(src, im_pad, top, bottom, left, right, cv::BORDER_DEFAULT);   //如果没法整除，进行补边操作；
    const int h = src.rows, w = src.cols;
    // Equalization

        int i = 0, j = 0, ni = m, nj = m;
        while (i < h)
        {
            ni = i + m;
            while (j < w)
            {
                nj = j + m;
                equalizeHist(im_pad.rowRange(i, ni).colRange(j, nj), im_pad.rowRange(i, ni).colRange(j, nj));
                j = nj;
            }
            i = ni;
            j = 0;
        }
        im_pad.copyTo(dst);       //copy




}

void My_AHE(Mat& src, Mat& dst, const int m)
{
    // Padding
    Mat src_pad;
    const int psz = m / 2;
    const int hend = src.rows + psz;
    const int wend = src.cols + psz;
    copyMakeBorder(src, src_pad, psz, psz, psz, psz, cv::BORDER_DEFAULT);  //扩展边界
    src.copyTo(dst);

    // Equalization
    int hist[256] = { 0 };
    float factor = 256.f / (m * m);
    int pi = 0, pj = 0, val = 0, cumv = 0;
    for (int i = psz; i < hend; i++)
    {
        for (int j = psz; j < wend; j++)
        {
            if (j == psz)       //只是第一列进行了窗口内的灰度值统计，后面采用了左出右进的加速算法；当然全部使用第一列这种普通算法也没问题，耗时较多。
            {
                // Reset
//                for (int pi = 0; pi < 256; pi++)
//                {
//                    hist[pi] = 0;
//                }

                // Statistical histogram
                for (pi = i - psz; pi <= i + psz; pi++)
                {
                    for (pj = j - psz; pj <= j + psz; pj++)
                    {
                        val = (int)src_pad.at<uchar>(pi, pj);
                        hist[val]++;
                    }
                }
            }
            else
            {
                // Pop left
                pj = j - psz - 1;
                for (pi = i - psz; pi <= i + psz; pi++)
                {
                    val = (int)src_pad.at<uchar>(pi, pj);
                    hist[val]--;
                }

                // Push right
                pj = j + psz;
                for (pi = i - psz; pi <= i + psz; pi++)
                {
                    val = (int)src_pad.at<uchar>(pi, pj);
                    hist[val]++;
                }   //这种加速算法仅需统计两列的像素值
            }

            // Equlization
            cumv = 0;
            val = (int)src_pad.at<uchar>(i, j);
            for (pi = 0; pi <= val; pi++)
            {
                cumv += hist[pi];           //计算直方图累计分布图
            }
            dst.at<uchar>(i - psz, j - psz) = uchar(cumv * factor);
        }
    }



}

void My_CHE(Mat& src, Mat& dst, const float clip_limit)
{
    // Statistical histogram
    const int h = src.rows;
    const int w = src.cols;
    int hist[256] = { 0 };
    uchar table[256] = { 0 };
    int idx = 0;
        for (int i = 0; i < src.rows; i++)
        {
            for (int j = 0; j < src.cols; j++)
            {
                idx = src.at<uchar>(i, j);
                hist[idx]++;
            }
        }
    // Limitation
    int steal = 0;
    const int limit = h * w * clip_limit / 256;
    for (int k = 0; k < 256; k++)
    {
        if (hist[k] > limit)
        {
            steal += hist[k] - limit;
            hist[k] = limit;
        }
    }
    // Hand out the steals averagely
    const int bonus = steal / 256;
    for (int k = 0; k < 256; k++)
    {
        hist[k] += bonus;   //平均分配到每一个灰度级上面
    }

    // Get mapping table
    float factor = 256.f / (h * w);
        int map_value, cumu_num = 0;
        for (int i = 0; i < 256; i++)
        {
            cumu_num += hist[i];
            map_value = int(cumu_num * factor);
//            if (map_value > 255) map_value = 255;
            table[i] = uchar(map_value);
        }

    // Map
        for (int i = 0; i < src.rows; i++)
        {
            for (int j = 0; j < src.cols; j++)
            {
                dst.at<uchar>(i, j) = table[(int)src.at<uchar>(i, j)];
            }
        }

}

void My_CLAHE(Mat& src, Mat& dst, const float clip_limit, const int m)
{
    // Padding
    Mat im_pad;
    src.copyTo(im_pad);
    const int h = src.rows, w = src.cols;

    //下面的代码是考虑图像不能被整齐划分的情况，对图像进行扩边操作
//    const int pdh = minInteger(h, m), pdw = minInteger(w, m);
//    const int res_h = pdh - h, res_w = pdw - w;
//    const int left = res_w / 2, top = res_h / 2;
//    const int right = res_w - left, bottom = res_h - top;
//    cv::copyMakeBorder(src, im_pad, top, bottom, left, right, cv::BORDER_DEFAULT);

    // Initialize histogram
    const int hnum = h / m;     //块的高度方向上的个数
    const int wnum = w / m;     //块的宽度方向上的个数
    const int bnum = hnum * wnum;   //块的个数
    int** hists = new int*[bnum];
    float** table = new float* [bnum];
    for (int i = 0; i < bnum; i++)
    {
        hists[i] = new int[256];
        table[i] = new float[256];
        for (int j = 0; j < 256; j++)
        {
            hists[i][j] = 0;
            table[i][j] = 0.f;
        }
    }

    // Statistical histogram, Get mapping table
    const int benum = m * m;
    const int limit = benum * clip_limit / 256;     //限制的灰度值
    int idx = 0, steal = 0, bonus = 0;
    for (int i = 0; i < hnum; i++)
    {
        for (int j = 0; j < wnum; j++)
        {
            // Statistical
            idx = i * wnum + j;
//            statHist(im_pad.rowRange(i * m, (i + 1) * m).colRange(j * m, (j + 1) * m), hists[idx]);
            int id = 0;
            for (int a = i * m; a < (i + 1) * m; a++)
            {
                for (int b = j * m; b < (j + 1) * m; b++)
                {
                    id = im_pad.at<uchar>(a, b);
                    hists[idx][id]++;
                }
            }
            // Limitation
            steal = 0;
            for (int k = 0; k < 256; k++)
            {
                if (hists[idx][k] > limit)
                {
                    steal += (hists[idx][k] - limit);
                    hists[idx][k] = limit;
                }
            }

            // Hand out the steals averagely
            bonus = steal / 256;
            for (int k = 0; k < 256; k++)
            {
                hists[idx][k] += bonus;
            }

            // Cumulative --> table
            table[idx][0] = hists[idx][0] * 255.f / benum;
            for (int k = 1; k < 256; k++)
            {
                table[idx][k] = table[idx][k - 1] + hists[idx][k] * 255.f / benum;

            }
//            for (int a = i * m; a < (i + 1) * m; a++)
//            {
//                for (int b = j * m; b < (j + 1) * m; b++)
//                {
//                    dst.at<uchar>(a, b) = table[idx][im_pad.at<uchar>(a, b)];
//                }
//            }
        }
    }
//    for (int j = 0; j < 256; j++)
//    {
//        delete[]hists[j];
//    }
    delete[]hists;

//     Equalization and Interpolation
    const int hm = m / 2;
    //im_pad.copyTo(im_pad);
    int hbi = 0, wbi = 0;
    int bidx[4] = { 0 };
    float p = 0.f, q = 0.f;
    for (int i = 0; i < h; i++)
    {
        for (int j = 0; j < w; j++)
        {
            //four coners 直接赋值
            if (i <= hm && j <= hm)
            {
                idx = 0;
                im_pad.at<uchar>(i, j) = (uchar)(table[idx][im_pad.at<uchar>(i, j)]);
            }
            else if (i <= hm && j >= w - hm)
            {
                idx = wnum - 1;
                im_pad.at<uchar>(i, j) = (uchar)(table[idx][im_pad.at<uchar>(i, j)]);
            }
            else if (i >= h - hm && j <= hm)
            {
                idx = bnum - wnum;
                im_pad.at<uchar>(i, j) = (uchar)(table[idx][im_pad.at<uchar>(i, j)]);
            }
            else if (i >= h - hm && j >= w - hm)
            {
                idx = bnum - 1;
                im_pad.at<uchar>(i, j) = (uchar)(table[idx][im_pad.at<uchar>(i, j)]);
            }
            //four edges except coners -- linear interpolation
            else if (i <= hm)
            {
                // hbi = 0;
                wbi = (j - hm) / m;
                bidx[0] = wbi;
                bidx[1] = bidx[0] + 1;
                p = (float)(j - (wbi * m + hm)) / m;
                q = 1 - p;
                im_pad.at<uchar>(i, j) = (uchar)(q * table[bidx[0]][im_pad.at<uchar>(i, j)] + p * table[bidx[1]][im_pad.at<uchar>(i, j)]);
            }
            else if (i >= ((hnum - 1) * m + hm))
            {
                hbi = hnum - 1;
                wbi = (j - hm) / m;
                bidx[0] = hbi * wnum + wbi;
                bidx[1] = bidx[0] + 1;
                float p = (float)(j - (wbi * m + hm)) / m;
                float q = 1 - p;
                im_pad.at<uchar>(i, j) = (uchar)(q * table[bidx[0]][im_pad.at<uchar>(i, j)] + p * table[bidx[1]][im_pad.at<uchar>(i, j)]);
            }
            else if (j <= hm)
            {
                hbi = (i - hm) / m;
                //wbi = 0;
                bidx[0] = hbi * wnum;
                bidx[1] = bidx[0] + wnum;
                p = (float)(i - (hbi * m + hm)) / m;
                q = 1 - p;
                im_pad.at<uchar>(i, j) = (uchar)(q * table[bidx[0]][im_pad.at<uchar>(i, j)] + p * table[bidx[1]][im_pad.at<uchar>(i, j)]);
            }
            else if (j >= ((wnum - 1) * m + hm))
            {
                hbi = (i - hm) / m;
                wbi = wnum - 1;
                bidx[0] = hbi * wnum + wbi;
                bidx[1] = bidx[0] + wnum;
                p = (float)(i - (hbi * m + hm)) / m;
                q = 1 - p;
                im_pad.at<uchar>(i, j) = (uchar)(q * table[bidx[0]][im_pad.at<uchar>(i, j)] + p * table[bidx[1]][im_pad.at<uchar>(i, j)]);
            }
            // Double linear interpolation
            else
            {
                hbi = (i - hm) / m;
                wbi = (j - hm) / m;
                bidx[0] = hbi * wnum + wbi;
                bidx[1] = bidx[0] + 1;
                bidx[2] = bidx[0] + wnum;
                bidx[3] = bidx[1] + wnum;
                p = (float)(i - (hbi * m + hm)) / m;
                q = (float)(j - (wbi * m + hm)) / m;
                im_pad.at<uchar>(i, j) = (uchar)(
                    (1 - p) * (1 - q) * table[bidx[0]][im_pad.at<uchar>(i, j)] +
                    (1 - p) * q * table[bidx[1]][im_pad.at<uchar>(i, j)] +
                    p * (1 - q) * table[bidx[2]][im_pad.at<uchar>(i, j)] +
                    p * q * table[bidx[3]][im_pad.at<uchar>(i, j)]);
            }
            im_pad.at<uchar>(i, j) = im_pad.at<uchar>(i, j) + (im_pad.at<uchar>(i, j) << 8) + (im_pad.at<uchar>(i, j) << 16);
        }
    }
    im_pad.copyTo(dst);

//    for (int j = 0; j < 256; j++)
//    {
//        delete[] table[j];
//    }
    delete[] table;



}

void show_histogram(Mat& img, Mat& Hist)
{
    //为计算直方图配置变量
    //首先是需要计算的图像的通道，就是需要计算图像的哪个通道（bgr空间需要确定计算 b或g货r空间）
    int channels = 0;
    //然后是配置输出的结果存储的 空间 ，用MatND类型来存储结果
    MatND dstHist;
    //接下来是直方图的每一个维度的 柱条的数目（就是将数值分组，共有多少组）
    int histSize[] = { 256 };       //如果这里写成int histSize = 256;   那么下面调用计算直方图的函数的时候，该变量需要写 &histSize
    //最后是确定每个维度的取值范围，就是横坐标的总数
    //首先得定义一个变量用来存储 单个维度的 数值的取值范围
    float midRanges[] = { 0, 256 };
    const float *ranges[] = { midRanges };

    calcHist(&img, 1, &channels, Mat(), dstHist, 1, histSize, ranges, true, false);

    //calcHist  函数调用结束后，dstHist变量中将储存了 直方图的信息  用dstHist的模版函数 at<Type>(i)得到第i个柱条的值
    //at<Type>(i, j)得到第i个并且第j个柱条的值

    //开始直观的显示直方图——绘制直方图
    //首先先创建一个黑底的图像，为了可以显示彩色，所以该绘制图像是一个8位的3通道图像
    Mat drawImage = Mat::zeros(Size(256, 256), CV_8UC3);
    //因为任何一个图像的某个像素的总个数，都有可能会有很多，会超出所定义的图像的尺寸，针对这种情况，先对个数进行范围的限制
    //先用 minMaxLoc函数来得到计算直方图后的像素的最大个数
    double g_dHistMaxValue;
    minMaxLoc(dstHist, 0, &g_dHistMaxValue, 0, 0);
    //将像素的个数整合到 图像的最大范围内
    //遍历直方图得到的数据
    for (int i = 0; i < 256; i++)
    {
        int value = cvRound(dstHist.at<float>(i) * 256 * 0.9 / g_dHistMaxValue);

        line(drawImage, Point(i, drawImage.rows - 1), Point(i, drawImage.rows - 1 - value), Scalar(255, 255, 255));
    }
    Hist = drawImage;
//    imshow("【原图直方图】", drawImage);
}
