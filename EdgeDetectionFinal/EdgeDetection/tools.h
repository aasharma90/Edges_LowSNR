#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

#ifndef TOOLS
#define TOOLS

void println(string str);
void println(int n);
int endRun(int exitCode);
Mat readImage(string img, bool kill = false);
void showImage(Mat& image, int fig, double scale = 1, bool wait = false);
double tic();
double toc(double start);
bool exists(const string& name);
int nchoosek(int n, int k);
int sub2ind(const int rows, const int cols, const int row, const int col);
void matSub2ind(const int rows, const int cols, const Mat& row, const Mat& col, Mat& dest);
void matSub2ind(const Size size, const Mat& row, const Mat& col, Mat& dest);
void ind2sub(const int ind, const int cols, const int rows, int &row, int &col);
int sign(int x);
void findIndices(const Mat& M, Mat& ind);
void copyIndices(const Mat& D, const Mat& ind, Mat& dest);
void setValueIfTrue(const double value, Mat& dst, const Mat& flag);
void setValueIfTrue(const Mat& src, Mat& dst, const Mat& flag);
void keepSelectedColumns(const Mat& src, const Mat& goodCols, Mat& dst);
void keepSelectedRows(const Mat& src, const Mat& goodCols, Mat& dst);
void keepTrue(const Mat& src, const Mat& keep, Mat& dst);
void setValuesInInd(const Mat& values, const Mat& ind, Mat& dst);
double maxValue(const Mat& m);
double minValue(const Mat& m);
void getHighestKValues(const Mat& src, Mat& dst, Mat& idx, int k);
void reorder(const Mat& idx, const Mat& src, Mat& dst);
void reorderCols(const Mat& idx, const Mat& src, Mat& dst);
int indToAngle(int rows, int cols, int ind0, int ind1);
void indToAngle(int rows, int cols, Mat& ind0, Mat& ind1, Mat& dst);
void mySplit(const Mat& I, Mat& I_split, char split_type, int part_no, int parts);
void myMerge(const Mat& I, Mat& E_split, Mat& E_merged, char split_type, int part_no, int parts);
void myThreshold(const Mat& E_merged_final, Mat& E_merged_final_thresholded, double thresh);
void structure_texture_decomposition_rof(const Mat& I, Mat& I_structure, Mat& I_texture, double theta, int nIter, double alp);
void myReshape(const Mat& I, Mat& I_reshaped, int new_rows, int new_cols);
void getLaplacian1(const Mat& I, Mat& W, Mat& A, Mat & consts, double epsilon, int win_size);
void scale_image(const Mat& I, Mat& I_scaled, double vlow, double vhigh, double ilow, double ihigh);
void displayMat(const Mat& I, int MaxRows, int MaxCols);
void waitUserKey(char key);
#endif

