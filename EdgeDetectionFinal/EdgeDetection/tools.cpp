#include <iostream>
#include <string>
#include <ctime>
#include <fstream>
#include <queue>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

const double PI = 3.141592653589793238463;

void println(string str){
	cout << str << endl;
}

void println(int n){
	cout << n << endl;
}

double tic(){
	return clock();
}

double toc(double start){
	double elapsed = (clock() - start)/1000.0;
	println(format("Elapsed %2.10f Seconds", elapsed));
	return elapsed;
}

int endRun(int exitCode){
	getchar();
	return exitCode;
}

int sign(int x){
	if (x > 0) return 1;
	else if(x < 0) return -1;
	else return 0;
}

int nchoosek(int n, int k)
{
	if (k == 0) return 1;
	if (n == 0) return 0;
	return nchoosek(n - 1, k - 1) + nchoosek(n - 1, k);
}

int sub2ind(const int rows, const int cols, const int row, const int col)
{
	return cols*row + col;
}

void matSub2ind(const int rows, const int cols, const Mat& row, const Mat& col, Mat& dest){
	assert(row.size == col.size);
	
	if (true){
		//Mat temp;
		//multiply(row, cols, temp);
		dest = cols*row + col;
	}
	else{
		dest = row.clone();
		assert(row.isContinuous() && col.isContinuous() && dest.isContinuous());

		double* p = (double*)row.data;
		double* cp = (double*)col.data;
		double* dp = (double*)dest.data;

		for (int i = 0; i < row.size().area(); ++i){
			*dp++ = sub2ind(rows, cols, (int)*p++, (int)*cp++);
		}
	}
}

void matSub2ind(const Size size, const Mat& row, const Mat& col, Mat& dest){
	matSub2ind(size.height, size.width, row, col, dest);
}

void ind2sub(const int ind, const int cols, const int rows, int &row, int &col)
{
	row = ind / cols;
	col = ind%cols;
}

bool exists(const string& name) {
	ifstream infile("thefile.txt");
	infile.close();
	return true;
}

Mat readImage(string img, bool kill = false){
	Mat image;
    image = imread(img, CV_LOAD_IMAGE_GRAYSCALE);
	
    if(! image.data )
    {
        println("Could not open or find the image");
		if(kill){
			exit(-1);
		}
    }
	return image;
}

void showImage(Mat& image, int fig, double scale = 1, bool wait = false){
	Mat iBig;
	resize(image, iBig, Size(0, 0), scale, scale);
	string winName = format("Window %d", fig);
	namedWindow( winName, WINDOW_AUTOSIZE );
    imshow( winName, iBig );
	if(wait){
		waitKey(0);
	}
}

void findIndices(const Mat& M, Mat& ind){
	assert(M.isContinuous());
	double* p = (double*)M.data;
	for (int i = 0; i < M.size().area(); ++i){
		if (*p++!= 0){
			ind.push_back(i);
		}
	}
	assert(ind.isContinuous());
	ind = ind.reshape(0, 1);
}

void copyIndices(const Mat& D, const Mat& ind, Mat& dest){
	Mat values = D.clone();
	dest = ind.clone();
	assert(values.isContinuous() && ind.isContinuous() && dest.isContinuous());
	values = values.reshape(0, 1);
	double* ip = (double*)ind.data;
	double* dp = (double*)dest.data;
	double* vp = (double*)values.data;

	for (int i = 0; i < ind.size().area(); ++i){
		int curInd = (int)*ip++;
		if (curInd >= 0){
			*dp++ = vp[curInd];
		}
		else{
			dp++;
		}
	}
}

void setValueIfTrue(const double value, Mat& dst, const Mat& flag){
	assert(dst.size().area() == flag.size().area());
	assert(dst.isContinuous() && flag.isContinuous());
	bool* fp = (bool*)flag.data;
	double* dp = (double*)dst.data;

	for (int i = 0; i < flag.size().area(); ++i){
		if (*fp++){
			*dp++ = value;
		}
		else{
			dp++;
		}
	}
}

void setValueIfTrue(const Mat& src, Mat& dst, const Mat& flag){
	assert(dst.size().area() == flag.size().area() && src.size().area() == flag.size().area());
	assert(src.isContinuous() && dst.isContinuous() && flag.isContinuous());
	bool* fp = (bool*)flag.data;
	double* dp = (double*)dst.data;
	double* sp = (double*)src.data;

	for (int i = 0; i < flag.size().area(); ++i){
		if (*fp++){
			*dp++ = *sp++;
		}
		else{
			dp++;
			sp++;
		}
	}
}

void keepSelectedRows(const Mat& src, const Mat& goodCols, Mat& dst){
	assert(src.rows == goodCols.cols && goodCols.rows == 1);
	assert(goodCols.isContinuous());

	Mat values = src.clone();
	Mat d;
	bool* p = (bool*)goodCols.data;

	for (int i = 0; i < goodCols.cols; ++i){
		if (*p++){
			d.push_back(values.row(i));
		}
	}
	d.copyTo(dst);
}

void keepSelectedColumns(const Mat& src, const Mat& goodCols, Mat& dst){
	assert(src.cols == goodCols.cols && goodCols.rows == 1);	
	assert(goodCols.isContinuous());

	Mat values = src.clone();
	Mat d;
	uchar* p = goodCols.data;

	for (int i = 0; i < goodCols.cols; ++i){
		if (*p++){
			d.push_back(values.col(i));
		}
	}
	d = d.reshape(0,values.rows);
	d.copyTo(dst);
}

void keepTrue(const Mat& src, const Mat& keep, Mat& dst){
	assert(src.size().area() == keep.size().area());
	assert(keep.isContinuous());
	Mat d;
	uchar* kp = keep.data;
	double* sp = (double*)src.data;
	for (int i = 0; i < keep.size().area(); ++i){
		double v = *sp++;
		if (*kp++){
			d.push_back(v);
		}
	}
	if (d.size().area()){
		d = d.reshape(0, 1);
		d.copyTo(dst);
	}
	else{
		dst.release();
	}
}

void setValuesInInd(const Mat& values, const Mat& ind, Mat& dst){
	assert(values.size().area() == ind.size().area());
	assert(values.isContinuous() && ind.isContinuous() && dst.isContinuous());

	double* vp = (double*)values.data;
	double* ip = (double*)ind.data;
	double* dp = (double*)dst.data;

	for (int i = 0; i < values.size().area(); ++i){
		dp[(int)*ip++] = *vp++;
	}
}

double maxValue(const Mat& m){
	double minVal;
	double maxVal;
	Point minLoc;
	Point maxLoc;
	minMaxLoc(m, &minVal, &maxVal, &minLoc, &maxLoc);
	return maxVal;
}

double minValue(const Mat& m) {
	double minVal;
	double maxVal;
	Point minLoc;
	Point maxLoc;
	minMaxLoc(m, &minVal, &maxVal, &minLoc, &maxLoc);
	return minVal;
}

void getHighestKValues(const Mat& src, Mat& dst,Mat& idx,int k){
	Mat values = src.clone();
	priority_queue<pair<double, double>> q;

	assert(values.isContinuous());
	double* p = (double*)values.data;

	for (int i = 0; i < values.size().area(); ++i) {
		q.push(pair<double, double>(*p++,(double)i));
	}
	for (int i = 0; i < k; ++i) {
		idx.push_back(q.top().second);
		dst.push_back(q.top().first);
		q.pop();
	}
}

void reorder(const Mat& idx, const Mat& src, Mat& dst){
	assert(idx.size().area() == src.size().area());
	assert(src.isContinuous() && idx.isContinuous());
	Mat d(src.size(), src.type());
	assert(d.isContinuous());
	int* ip = (int*)idx.data;
	double* dp = (double*)d.data;
	double* sp = (double*)src.data;

	for (int i = 0; i < idx.size().area(); ++i){
		*dp++ = sp[*ip++];
	}
	d.copyTo(dst);
}

void reorderCols(const Mat& idx, const Mat& src, Mat& dst){
	assert(idx.size().area() == src.cols);
	assert(idx.isContinuous());
	Mat d(src.size(), src.type());
	assert(d.isContinuous());
	int* ip = (int*)idx.data;
	
	for (int i = 0; i < idx.size().area(); ++i){
		src.col(*ip++).copyTo(d.col(i));
	}
	d.copyTo(dst);
}

int indToAngle(int rows, int cols, int ind0, int ind1){
	int x0, y0, x1, y1;
	ind2sub(ind0, rows, cols, x0, y0);
	ind2sub(ind1, rows, cols, x1, y1);
	int v1 = x1 - x0;
	int v2 = y1 - y0;

	double angle = (v1 == 0) ? sign(v2)*PI/2 : atan(v2 / v1);
	angle *= 180 / PI;
	int ang = (int)(angle)+360;

	if (v1 < 0){
		ang += 180;
	}
	ang = ang%360;

	assert(ang >= 0 && ang <360);
	return ang;
}

void indToAngle(int rows, int cols, Mat& ind0, Mat& ind1, Mat& dst){
	Mat d(ind0.size(), ind0.type());
	assert(ind1.isContinuous() && ind0.isContinuous() && d.isContinuous());
	double* dp = (double*)d.data;
	double* p0 = (double*)ind0.data;
	double* p1 = (double*)ind1.data;

	for (int i = 0; i < d.size().area(); ++i){
		*dp++ = indToAngle(rows, cols, (int)*p0++, (int)*p1++);
	}
	d.copyTo(dst);
}

void mySplit(const Mat& I, Mat& I_split, char split_type, int part_no, int parts)
{
	int i;
	double new_rows, new_cols;
	i = part_no;
	Range rx, ry;
	switch (split_type)
	{
	case 'h': new_rows = I.rows / parts;
		new_rows = new_rows == floor(new_rows) ? new_rows : floor(new_rows);
		new_cols = double(I.cols);
		break;
	case 'v': new_cols = I.cols / parts;
		new_cols = new_cols == floor(new_cols) ? new_cols : floor(new_cols);
		new_rows = double(I.rows);
		break;
	}
	switch (split_type)
	{
	case 'h':
		if (i == parts)
		{
			// Compensate for any missed row in the last split
			ry = Range((i - 1)*new_rows, I.rows);
		}
		else
		{
			ry = Range((i - 1)*new_rows, i*new_rows);
		}
		rx = Range(0, new_cols);
		break;
	case 'v':
		if (i == parts)
		{
			// Compensate for any missed col in the last split
			rx = Range((i - 1)*new_cols, I.cols);
		}
		else
		{
			rx = Range((i - 1)*new_cols, i*new_cols);
		}
		ry = Range(0, new_rows);
		break;
	}
	//cout << i << ", " << rx.end << ", " << ry.end << endl;
	I_split = I(ry, rx);
}

void myMerge(const Mat& I, Mat& E_split, Mat& E_merged, char split_type, int part_no, int parts)
{
	int i;
	double new_rows, new_cols, original_rows, original_cols;
	i = part_no;
	original_rows = double(I.rows);
	original_cols = double(I.cols);
	Range rx, ry;
	switch (split_type)
	{
	case 'h':
		new_rows = original_rows / parts;
		new_rows = new_rows == floor(new_rows) ? new_rows : floor(new_rows);
		new_cols = original_cols;
		break;
	case 'v':
		new_cols = original_cols / parts;
		new_cols = new_cols == floor(new_cols) ? new_cols : floor(new_cols);
		new_rows = original_rows;
		break;
	}
	switch (split_type)
	{
	case 'h':
		if (i == parts)
		{
			// Compensate for any missed row in the last split
			ry = Range((i - 1)*new_rows, I.rows);
		}
		else
		{
			ry = Range((i - 1)*new_rows, i*new_rows);
		}
		rx = Range(0, new_cols);
		break;
	case 'v':
		if (i == parts)
		{
			// Compensate for any missed col in the last split
			rx = Range((i - 1)*new_cols, I.cols);
		}
		else
		{
			rx = Range((i - 1)*new_cols, i*new_cols);
		}
		ry = Range(0, new_rows);
		break;
	}
	//cout << E_split.rows << ", " << E_split.cols << endl;
	//cout << E_merged.rows << ", " << E_merged.cols << endl;
	//cout << "myMerge - Part : " << i << ", rx : (" << rx.start <<  ", " << rx.end << "), ry : (" << ry.start << ", " << ry.end << ")" << endl;
	E_split.copyTo(E_merged(ry, rx));
	//showImage(E_split, 100, 4, false);
	//showImage(E_merged, 101, 4, true);
}

void myThreshold(const Mat& E_merged_final, Mat& E_merged_final_thresholded, double thresh)
{
	int rows = E_merged_final.rows;
	int cols = E_merged_final.cols;
	for (int r_i = 0; r_i < rows; r_i++)
	{
		for (int c_i = 0; c_i < cols; c_i++)
		{
			//cout << E_merged_final_thresholded.at<double>(r_i, c_i) << endl;
			if (E_merged_final_thresholded.at<double>(r_i, c_i) < thresh)
			{
				E_merged_final_thresholded.at<double>(r_i, c_i) = 0;
			}
		}
	}
}


void scale_image(const Mat& I, Mat& I_scaled, double vlow, double vhigh, double ilow, double ihigh)
{
	I_scaled = (I - ilow) / (ihigh - ilow) * (vhigh - vlow) + vlow;
}

void displayMat(const Mat& I, int MaxRows, int MaxCols)
{
	int rows = (I.rows > MaxRows) ? MaxRows : I.rows;
	int cols = (I.cols > MaxCols) ? MaxCols : I.cols;
	cout << "Matrix of size " << I.rows << "x" << I.cols << ", Displaying " << rows << "x" << cols << endl;
	cout << "[";
	for (int r_i = 0; r_i < rows; r_i++)
	{
		for (int c_i = 0; c_i < cols; c_i++)
		{
			cout << I.at<double>(r_i, c_i) << ",";
		}
		cout << endl;
	}
	cout << "]" << endl << endl;
}

void waitUserKey(char key)
{
	char user_key;
	cout << endl << "To move forward, press '" << key << "' key (then hit 'Enter')" << endl;
	cin >> user_key;
	while (user_key != key)
	{
		// Wait;
	}
}
void myReshape(const Mat& I, Mat& I_reshaped, int new_rows, int new_cols)
{
	if (new_rows*new_cols != I.rows*I.cols)
	{
		cerr << "Error! No of elements should not change for reshaping" << endl;
	}
	Mat I_temp = Mat(1, I.rows*I.cols, CV_64F, double(0));
	int count = 0;
	for (int r_i = 0; r_i < I.rows; r_i++)
	{
		for (int c_i = 0; c_i < I.cols; c_i++)
		{
			count = count + 1;
			I_temp.at<double>(0, count-1) = I.at<double>(c_i, r_i);
		}
	}
	count = 0;
	for (int r_i = 0; r_i < new_rows; r_i++)
	{
		for (int c_i = 0; c_i < new_cols; c_i++)
		{
			count = count + 1;
			I_reshaped.at<double>(r_i, c_i) = I_temp.at<double>(0, count-1);
		}
	}
}

/* Code adapted from structure_texture_decomposition_rof
MATALB code provided by 
%   Author: Deqing Sun, Department of Computer Science, Brown University
%   Contact: dqsun@cs.brown.edu
*/
void structure_texture_decomposition_rof(const Mat& I, Mat& I_structure, Mat& I_texture, double theta, int nIter, double alp)
{
	cout << "Performing S-T ROF decomposition (Max. Iterations = " << nIter << ")" << endl;
	Mat I_scaled, I_copy;
	Mat	p1, p2, p1_sqaured, p2_squared, p_mag, p_reprojection;
	Mat div_p1, div_p2, div_p;
	Mat I_x, I_y;
	Mat I_structure_temp, I_texture_temp;
	// Scale Image to [-1, 1]
	double max_I = maxValue(I);
	double min_I = minValue(I);
	//cout << min_I << ", " << max_I << endl;
	scale_image(I, I_scaled, -1, 1, min_I, max_I);
	max_I = maxValue(I_scaled);
	min_I = minValue(I_scaled);
	//cout << min_I << ", " << max_I << endl;	

	// Backup image
	I_scaled.copyTo(I_copy);

	// Step Size
	double delta = 1 / (4*theta);

	// Initialize dual variable p to be 0
	p1 = Mat(I.rows, I.cols, CV_64F, double(0));
	p2 = Mat(I.rows, I.cols, CV_64F, double(0));


	// Gradient Descent 
	int iter;
	Mat magic4 =  (Mat_<double>(4, 4) << 16, 2, 3, 13, 5, 11, 10, 8, 9, 7, 6, 12, 4, 14, 15, 1);
	Mat filter1 = (Mat_<double>(1, 3) << -1, 1, 0);
	Mat filter2 = (Mat_<double>(3, 1) << -1, 1, 0);
	Mat filter3 = (Mat_<double>(1, 2) << -1, 1);
	Mat filter4 = (Mat_<double>(2, 1) << -1, 1);
	//Mat magic4_out;
	//filter2D(magic4, magic4_out, -1, filter1, Point(-1, -1), 0.0, BORDER_CONSTANT);
	//displayMat(magic4_out);
	//filter2D(magic4, magic4_out, -1, filter2, Point(0, 0), 0.0, BORDER_REPLICATE);
	//displayMat(magic4_out);
	//filter2D(magic4, magic4_out, -1, filter3, Point(0, 0), 0.0, BORDER_REPLICATE);
	//displayMat(magic4_out);
	I_structure_temp = I_scaled;
	for (iter = 1; iter <= nIter; iter++)
	{

		// Compute divergence
		filter2D(p1, div_p1, -1, filter1, Point(-1, -1), 0.0, BORDER_CONSTANT);
		filter2D(p2, div_p2, -1, filter2, Point(-1, -1), 0.0, BORDER_CONSTANT);
		div_p = div_p1 + div_p2;
		//displayMat(div_p1, 4, 4);
		//displayMat(div_p2, 4, 4);
		//displayMat(div_p, 4, 4);


		// Compute gradients
		filter2D(I_scaled + theta*div_p, I_x, -1, filter3, Point(0, 0), 0.0, BORDER_REPLICATE); // Make the anchor point (0,0) to match MATLAB results !
		filter2D(I_scaled + theta*div_p, I_y, -1, filter4, Point(0, 0), 0.0, BORDER_REPLICATE); // Make the anchor point (0,0) to match MATLAB results !

		// Update dual variables
		p1 = p1 + delta*(I_x);
		p2 = p2 + delta*(I_y);

		// Re-projection to p_mag <= 1
		pow(p1, 2, p1_sqaured);
		pow(p2, 2, p2_squared);
		pow((p1_sqaured + p2_squared), 0.5, p_mag);
		p_reprojection = max(p_mag, 1);
		divide(p1, p_reprojection, p1);
		divide(p2, p_reprojection, p2);
		//displayMat(p1, 4, 4);
	}

	// Re-compute divergence
	filter2D(p1, div_p1, -1, filter1, Point(-1, -1), 0.0, BORDER_CONSTANT);
	filter2D(p2, div_p2, -1, filter2, Point(-1, -1), 0.0, BORDER_CONSTANT);
	div_p = div_p1 + div_p2;
	//displayMat(div_p, 4, 4);

	// Compute Structure component
	I_structure_temp = I_scaled + theta*div_p;
	//displayMat(I_structure_temp, 4, 4);

	// Compute Texture component
	I_texture_temp = I_copy - alp*I_structure_temp;
	//displayMat(I_texture_temp, 4, 4);

	// Scale the components to [0-1]
	double max_I_texture_temp = maxValue(I_texture_temp);
	double min_I_texture_temp = minValue(I_texture_temp);
	scale_image(I_texture_temp, I_texture, 0, 1, min_I_texture_temp, max_I_texture_temp);
	double max_I_structure_temp = maxValue(I_structure_temp);
	double min_I_structure_temp = minValue(I_structure_temp);
	scale_image(I_structure_temp, I_structure, 0, 1, min_I_structure_temp, max_I_structure_temp);
	cout << "S-T ROF decomposition complete!" << endl;
}

/* Code adapted from getLaplacian1
MATALB code provided by
%   Author: Unknown
%   Contact: Unknown
*/
void getLaplacian1(const Mat& I, Mat& W, Mat& A, Mat& consts, double epsilon, int win_size)
{
	int neb_size = pow((win_size * 2 + 1),2);
	int h = I.rows;
	int w = I.cols;
	int img_size = h*w;
	cout << "Generating Laplacian Affnity Matrix of size " << img_size << "x" << img_size << endl;
	// Skipping the imerode function for now
	// consts=imerode(consts,ones(win_size*2+1));

	// Generate linear indices matrix in hxw form
	Mat indsM = Mat(h, w, CV_64F, double(0));
	int index = 0;
	for (int r_i = 0; r_i < indsM.rows; r_i++)
	{
		for (int c_i = 0; c_i < indsM.cols; c_i++)
		{   
			index++;
			indsM.at<double>(c_i, r_i) = index;
		}
	}
	//displayMat(indsM, h, w);
	
	// Compute tlen and initialize row_inds, col_inds, vals
	Mat consts_submat = consts(Range(win_size, consts.rows - win_size), Range(win_size, consts.cols - win_size));
	int tlen = sum(sum(1 - consts_submat)).val[0] * pow(neb_size, 2);
	//cout << tlen << endl;
	Mat row_inds = Mat(tlen, 1, CV_64F, double(0));
	Mat col_inds = Mat(tlen, 1, CV_64F, double(0));
	Mat vals     = Mat(tlen, 1, CV_64F, double(0));

	int len = 0;
	// Compute vals
	for (int c_i = win_size; c_i < w - win_size; c_i++)
	{
		for (int r_i = win_size; r_i < h - win_size; r_i++)
		{
			// Compute win_inds
			Mat win_inds = indsM(Range(r_i - win_size, r_i + 1 + win_size), Range(c_i - win_size, c_i + 1 + win_size));
			Mat win_inds_reshaped = Mat(win_inds.rows*win_inds.cols, 1, CV_64F, double(0));
			Mat win_inds_transpose;
			//displayMat(win_inds, win_inds.rows, win_inds.cols);
			myReshape(win_inds, win_inds_reshaped, win_inds.rows*win_inds.cols, 1);
			win_inds_reshaped.copyTo(win_inds);
			//displayMat(win_inds, win_inds.rows, win_inds.cols);

			// Compute winI
			Mat winI = I(Range(r_i - win_size, r_i + 1 + win_size), Range(c_i - win_size, c_i + 1 + win_size));
			Mat winI_reshaped = Mat(winI.rows*winI.cols, 1, CV_64F, double(0));
			//displayMat(winI, winI.rows, winI.cols);
			myReshape(winI, winI_reshaped, neb_size, 1);
			winI_reshaped.copyTo(winI);
			//displayMat(winI, winI.rows, winI.cols);

			// Compute win_mu and win_var
			Mat winI_transpose, data_temp;
			double win_mu, win_var;
			win_mu = mean(winI).val[0];		
			transpose(winI, winI_transpose);
			data_temp = winI_transpose * winI / neb_size - win_mu*win_mu + epsilon / neb_size;
			invert(data_temp, data_temp);
			win_var = data_temp.at<double>(0, 0);
			//cout << win_mu << "," << win_var << endl;

			// Compute tvals
			repeat(win_mu, neb_size, 1, data_temp);
			winI = winI - data_temp;
			transpose(winI, winI_transpose);
		    Mat tvals = (1 + winI*win_var*winI_transpose)/neb_size;
			//displayMat(winI, winI.rows, winI.cols);
			//displayMat(tvals, tvals.rows, tvals.cols);

			// Compute row_inds and col_inds
			Mat data_temp_reshaped = Mat(pow(neb_size, 2), 1, CV_64F, double(0));
			repeat(win_inds, 1, neb_size, data_temp);
			myReshape(data_temp, data_temp_reshaped, pow(neb_size, 2), 1);
			//displayMat(data_temp_reshaped, data_temp_reshaped.rows, data_temp_reshaped.cols);
			data_temp_reshaped.copyTo(row_inds(Range(len, pow(neb_size, 2) + len), Range(0,1)));
			//displayMat(row_inds, row_inds.rows, row_inds.cols);
			transpose(win_inds, win_inds_transpose);
			repeat(win_inds_transpose, neb_size, 1, data_temp);
			myReshape(data_temp, data_temp_reshaped, pow(neb_size, 2), 1);
			//displayMat(data_temp_reshaped, data_temp_reshaped.rows, data_temp_reshaped.cols);
			data_temp_reshaped.copyTo(col_inds(Range(len, pow(neb_size, 2) + len), Range(0, 1)));
			//displayMat(col_inds, col_inds.rows, col_inds.cols);

			// Compute vals
			Mat tvals_reshaped = Mat(tvals.rows*tvals.cols, 1, CV_64F, double(0));
			myReshape(tvals, tvals_reshaped, tvals.rows*tvals.cols, 1);
			tvals_reshaped.copyTo(vals(Range(len, pow(neb_size, 2) + len), Range(0, 1)));
			//displayMat(vals, vals.rows, vals.cols);

			// Change len
			len = len + pow(neb_size, 2);
		}
	}
	vals(Range(0, len), Range(0, 1)).copyTo(vals);
	row_inds(Range(0, len), Range(0, 1)).copyTo(row_inds);
	col_inds(Range(0, len), Range(0, 1)).copyTo(col_inds);
	Mat row_inds_last_10 = row_inds(Range(row_inds.rows - 10, row_inds.rows), Range(0, 1));
	Mat col_inds_last_10 = col_inds(Range(col_inds.rows - 10, col_inds.rows), Range(0, 1));
	Mat vals_last_10 = vals(Range(vals.rows - 10, vals.rows), Range(0, 1));
	//displayMat(vals, vals.rows, vals.cols);
	//displayMat(row_inds_last_10, 10, 1);
	//displayMat(col_inds_last_10, 10, 1);
	//displayMat(vals_last_10, 10, 1);
	//cout << vals.rows << ", " << vals.cols << endl;

	// Compute the Laplacian affinity weight matrix
	int count = 0;
	int row_index, col_index;
	for (count = 1;count <= len;count++)
	{
		row_index = row_inds.at<double>(count - 1, 0);
		col_index = col_inds.at<double>(count - 1, 0);
		double data = vals.at<double>(count - 1, 0);
		//cout << "(" << row_index << ", " << col_index << ")" << "=" << data << endl;
		W.at<double>(row_index - 1, col_index - 1) = W.at<double>(row_index - 1, col_index - 1) + data;
	}
	//displayMat(W, 5, 5);
	Mat W_col_sum = Mat(1, W.cols, CV_64F, double(0));
	reduce(W, W_col_sum, 0, CV_REDUCE_SUM);
	
	// Compute the Laplacian matrix
	for (int r_i = 0;r_i < W.rows;r_i++)
	{
		for (int c_i = 0;c_i < W.cols;c_i++)
		{
			if (r_i == c_i)
				A.at<double>(r_i, c_i) = W_col_sum.at<double>(0, c_i) - W.at<double>(r_i, c_i);
			else
				A.at<double>(r_i, c_i) = -W.at<double>(r_i, c_i);
		}
	}
	//displayMat(A, 5, 5);
}

