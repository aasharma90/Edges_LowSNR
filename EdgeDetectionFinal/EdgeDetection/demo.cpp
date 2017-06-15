#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "tools.h"
#include "Detector.h"
#include <thread>
#include <stdio.h>
#include <direct.h>
//#include "mex.h"

using namespace std;
using namespace cv;

void myRunIm(const Mat& I, Mat& E, MyParam& prm);
void myRunIm_experiment(const Mat& I, Mat& E, MyParam& prm);
void myWrapper(const Mat& I, Mat& E, const MyParam& prm, const Range& ry, const Range& rx);

int main( int argc, char** argv )
{

	// For Noisy Image Demo, set run_demo_image = 1
	int run_demo_image = 1;
	// To perform splitting and merging, set split_and_merge = 1
	int split_and_merge = 1;
	// To perform threshold before adding the merged resutls
	int threshold_before_adding_merged = 0;
	// To involve structure-texture decomposition 
	int perform_st_decomposition = 0;
	// To generate Laplacian Affnity matrix
	int generate_laplacian_affinity_matrix = 0;
	// Experiment with new response function
	int experiment_with_new_response_function = 0;


	Mat I;
	MyParam prm;

	if (experiment_with_new_response_function == 1)
	{
		// Read image
		I = readImage("Simulations/myCurves2.png");
		I.convertTo(I, TYPE);
		I = I / 255;
		// Patch where false postives are detected
		Mat Patch;
		Patch = I(Range(0, 15), Range(110, 125));
		showImage(Patch, 1, 10, false);
		Mat E_patch;
		prm.slidingWindow = 0;
		prm.noisyImage = true;
		prm.parallel = false;
		prm.splitPoints = 0;
		//prm.generate_Laplacian = true;
		//prm.generate_Laplacian_win_size = 4;
		//prm.interpolation = false;
		// First Iteration, all Image
		myRunIm_experiment(Patch, E_patch, prm);
		E_patch = 1 - E_patch;
		// Show the edge map
		displayMat(E_patch, E_patch.rows, E_patch.cols);
		showImage(E_patch, 2, 10, true);
		waitUserKey('y');
	}


	if (generate_laplacian_affinity_matrix == 1)
	{
		I = readImage("Simulations/myCurves2.png");
		I.convertTo(I, TYPE);
		I = I / 255;
		Mat magic4 = (Mat_<double>(4, 4) << 16, 2, 3, 13, 5, 11, 10, 8, 9, 7, 6, 12, 4, 14, 15, 1);
		
		int h = I.rows;
		int w = I.cols;
		int img_size = h*w;
		Mat W = Mat(img_size, img_size, CV_64F, double(0));
		Mat A = Mat(img_size, img_size, CV_64F, double(0));	
		double epsilon = 0.0000001;
		int win_size = 3;
		Mat consts = Mat(h, w, TYPE, ZERO);
		getLaplacian1(I, W, A, consts, epsilon, win_size);
		while (1);
	}

	if (run_demo_image)
	{
		//Demo for demo noisy image
		cout << "Noisy Image Demo:" << endl;
		I = readImage("Simulations/myCurves2.png");
		I.convertTo(I, TYPE);
		I = I / 255;
		cout << "Image size of " << I.rows << " x " << I.cols << endl;
		// Show the noisy image
		showImage(I, 1, 4, false);
		_sleep(100);
		// Generate Edge Map
		Mat E;
		prm.slidingWindow = 0;
		prm.noisyImage = true;
		prm.parallel = false;
		prm.splitPoints = 0;
		//prm.interpolation = false;
		// First Iteration, all Image
		myRunIm(I, E, prm);
		E = 1 - E;
		// Show the edge map
		showImage(E, 2, 4, split_and_merge ? false : (perform_st_decomposition ? false : true));
		E = E * 255;
		E.convertTo(E, CV_8U);
		imwrite("Simulations/my_RESULT1.png", E);

		// Check whether or not to perform ST-ROF
		if (perform_st_decomposition)
		{
			// Perform sturcture-texture decomposition
			double theta_array[] = { 0.125, 0.25};
			int nIter  = 200;
			double alp = 0.95; // 20:1 decomposition
			int theta_array_size = end(theta_array) - begin(theta_array);
			Mat E_structure_merged;
			E.convertTo(E, TYPE);
			E = E / 255;
			E.copyTo(E_structure_merged); // Start with the original edge map
			for (int theta_array_c = 0; theta_array_c < theta_array_size; theta_array_c++)
			{
				Mat I_structure, I_texture, E_structure;
				structure_texture_decomposition_rof(I, I_structure, I_texture, theta_array[theta_array_c], nIter, alp);
				myRunIm(I_structure, E_structure, prm);
				E_structure = 1 - E_structure;
				showImage(E_structure, 500 + theta_array_c, 4, false);
				E_structure_merged = E_structure_merged + E_structure;
			}
			// Combine all the results ('+ 1' for the edge map on the original image)
			E_structure_merged = E_structure_merged / (theta_array_size + 1);
			// Update E
			E_structure_merged.copyTo(E);
			// Show the new edge map
			showImage(E, 600, 4, split_and_merge ? false : true);
			E = E * 255;
			E.convertTo(E, CV_8U);
			imwrite("Simulations/my_RESULT1_with_ST_dec.png", E);
		}

	}
	else
	{
		//Real Image Demo
		cout << "Real Image Demo:" << endl;
		I = readImage("real/night1_left_LIME.png");
		I.convertTo(I, TYPE);
		I = I / 255;
		// Show the noisy image
		showImage(I, 1, 1, false);
		// Generate Edge Map
		Mat E;
		prm.slidingWindow = 129;
		prm.noisyImage = true;
		prm.parallel = true;
		prm.splitPoints = 0;
		// First Iteration, all Image
		myRunIm(I, E, prm);
		E = 1 - E;
		// Show the Edge Map
		showImage(E, 2, 1, split_and_merge ? false : true);
		E = E * 255;
		E.convertTo(E, CV_8U);
		imwrite("Real/my_RESULT1.png", E);

		// Check whether or not to perform ST-ROF
		if (perform_st_decomposition)
		{
			// Perform sturcture-texture decomposition
			double theta_array[] = { 0.125, 0.25 };
			int nIter = 200;
			double alp = 0.95; // 20:1 decomposition
			int theta_array_size = end(theta_array) - begin(theta_array);
			Mat E_structure_merged;
			E.convertTo(E, TYPE);
			E = E / 255;
			E.copyTo(E_structure_merged); // Start with the original edge map
			for (int theta_array_c = 0; theta_array_c < theta_array_size; theta_array_c++)
			{
				Mat I_structure, I_texture, E_structure;
				structure_texture_decomposition_rof(I, I_structure, I_texture, theta_array[theta_array_c], nIter, alp);
				myRunIm(I_structure, E_structure, prm);
				E_structure = 1 - E_structure;
				//showImage(E_structure, 500 + theta_array_c, 1, false);
				E_structure_merged = E_structure_merged + E_structure;
			}
			// Combine all the results ('+ 1' for the edge map on the original image)
			E_structure_merged = E_structure_merged / (theta_array_size + 1);
			// Update E
			E_structure_merged.copyTo(E);
			// Show the new edge map
			showImage(E, 600, 1, split_and_merge ? false : true);
			E = E * 255;
			E.convertTo(E, CV_8U);
			imwrite("Real/my_RESULT1_with_ST_dec.png", E);
		}

	}

	if (split_and_merge == 1)
	{ // Start of split_and_merge
		// Check affect of splitting 
		Mat E_merged = Mat(I.rows, I.cols, TYPE, ZERO);;
		Mat E_merged_final = Mat(I.rows, I.cols, TYPE, ZERO);
		Mat E_merged_thresholded = Mat(I.rows, I.cols, TYPE, ZERO);;
		Mat E_merged_final_thresholded = Mat(I.rows, I.cols, TYPE, ZERO);
		int parts_array[] = { 2, 3, 4, 5, 6, 7, 8, 9 };
		char split_type_array[] = { 'v', 'h' };
		char split_type;
		int part_no, parts, no_of_runs;
		no_of_runs = 0;
		for (int split_type_c = 0; split_type_c < 2; split_type_c++)
		{
			split_type = split_type_array[split_type_c];
			for (int parts_c = 0; parts_c < end(parts_array) - begin(parts_array); parts_c++)
			{
				no_of_runs++;
				parts = parts_array[parts_c];
				for (int i = 1;i <= parts;i++)
				{
					Mat I_split, E_split;
					part_no = i;
					mySplit(I, I_split, split_type, part_no, parts);
					myRunIm(I_split, E_split, prm);
					E_split = 1 - E_split;
					//cout << "Output Split Image size of " << E_split.rows << " x " << E_split.cols << endl;
					//showImage(E_split, i+parts, 4, false);
					if (perform_st_decomposition)
					{
						// Perform sturcture-texture decomposition
						double theta_array[] = { 0.125, 0.25 };
						int nIter = 200;
						double alp = 0.95; // 20:1 decomposition
						int theta_array_size = end(theta_array) - begin(theta_array);
						Mat E_split_structure_merged;
						//E_split.convertTo(E_split, TYPE);
						//E_split = E_split / 255;
						E_split.copyTo(E_split_structure_merged); // Start with the original edge map
						for (int theta_array_c = 0; theta_array_c < theta_array_size; theta_array_c++)
						{
							Mat I_split_structure, I_split_texture, E_split_structure;
							structure_texture_decomposition_rof(I_split, I_split_structure, I_split_texture, theta_array[theta_array_c], nIter, alp);
							myRunIm(I_split_structure, E_split_structure, prm);
							E_split_structure = 1 - E_split_structure;
							//showImage(E_split_structure, 700 + theta_array_c, 4, false);
							E_split_structure_merged = E_split_structure_merged + E_split_structure;
						}
						// Combine all the results ('+ 1' for the edge map on the original image)
						E_split_structure_merged = E_split_structure_merged / (theta_array_size + 1);
						// Update E
						E_split_structure_merged.copyTo(E_split);
						// Show the new edge map
						// showImage(E_split, 600, 4, split_and_merge ? false : true);
						// E_split = E_split * 255;
						// E_split.convertTo(E_split, CV_8U);
						// imwrite("Real/my_RESULT1_with_ST_dec.png", E_split);
					}
					myMerge(I, E_split, E_merged, split_type, part_no, parts);
					//showImage(E_merged, i+2*parts, 4, false);
				}
				// Show the edge map
				//showImage(E_merged, 3 + parts_c, 4, false);
				if (threshold_before_adding_merged == 1)
				{
					// Threshold the image
					E_merged = 1 - E_merged;
					double thresh = 0.2;
					E_merged.copyTo(E_merged_thresholded);
					myThreshold(E_merged, E_merged_thresholded, thresh);
					E_merged_thresholded = 1 - E_merged_thresholded;
					add(E_merged_final, E_merged_thresholded, E_merged_final);
				}
				else
				{
					add(E_merged_final, E_merged, E_merged_final);
				}
			}
		}
		E_merged_final = E_merged_final / no_of_runs;
		// Threshold the image
		E_merged_final = 1 - E_merged_final;
		double thresh = 0.2;
		E_merged_final.copyTo(E_merged_final_thresholded);
		myThreshold(E_merged_final, E_merged_final_thresholded, thresh);
		// Display results
		E_merged_final = 1 - E_merged_final;
		E_merged_final_thresholded = 1 - E_merged_final_thresholded;
		showImage(E_merged_final, 100, run_demo_image ? 4 : 1, false);
		showImage(E_merged_final_thresholded, 101, run_demo_image ? 4 : 1, false);
		E_merged_final = E_merged_final * 255;
		E_merged_final.convertTo(E_merged_final, CV_8U);
		E_merged_final_thresholded = E_merged_final_thresholded * 255;
		E_merged_final_thresholded.convertTo(E_merged_final_thresholded, CV_8U);
		if (perform_st_decomposition)
		{
			imwrite(run_demo_image ? "Simulations/my_RESULT2_with_ST_dec.png" : "Real/my_RESULT2_with_ST_dec.png", E_merged_final);
			imwrite(run_demo_image ? "Simulations/my_RESULT3_with_ST_dec.png" : "Real/my_RESULT3_with_ST_dec.png", E_merged_final_thresholded);
		}
		else
		{
			imwrite(run_demo_image ? "Simulations/my_RESULT2.png" : "Real/my_RESULT2.png", E_merged_final);
			imwrite(run_demo_image ? "Simulations/my_RESULT3.png" : "Real/my_RESULT3.png", E_merged_final_thresholded);
		}
		
	} // end of split_and_merge
	
	cout << "Simulations Finished ! Close the window to exit !" << endl;
	waitKey(0);
	println("Finished");
	return 0;
}


std::mutex E_mutex;

void myRunIm_experiment(const Mat& I, Mat& E, MyParam& prm) {
	if (!prm.slidingWindow) {
		E = Mat(I.rows, I.cols, TYPE, ZERO);
		Detector d(I, prm);
		d._my_experiment = 1;
		Mat curE = d.runIm();
		E_mutex.lock();
		E(Range::all(), Range::all()) = max(E(Range::all(), Range::all()), curE);
		E_mutex.unlock();
	}
	if (maxValue(E) != 0)
	{
		E = E / maxValue(E);
	}
}

void myRunIm(const Mat& I, Mat& E, MyParam& prm){
	if (!prm.slidingWindow){
		E = Mat(I.rows, I.cols, TYPE, ZERO);
		myWrapper(I, E, prm, Range::all(), Range::all());
	}
	else{
		prm.parallel = true;
		prm.printToScreen = false;
		int s = min(I.cols, I.rows);
		double j = log2(s);
		j = j == floor(j) ? floor(j) - 1 : floor(j);
		s = (int)pow(2,j) + 1;
		s = min(s, (int)prm.slidingWindow);
		E = Mat(I.rows, I.cols, TYPE, ZERO);
		int ds = (s - 1) / 2;
		Range rx, ry;
		double start = tic();
		int ITER = 0;
		cout << (I.cols/ds+1)*(I.rows/ds+1) << " ITERATIONS" << endl;
		cout << s << " BLOCK" << endl;
		vector<thread> tasks;
		bool parallel = false;
		for (int x = 0; x < I.cols; x += ds){
			for (int y = 0; y < I.rows; y += ds){
				rx = x + s >= I.cols ? Range(I.cols - s, I.cols) : Range(x, x + s);
				ry = y + s >= I.rows ? Range(I.rows - s, I.rows) : Range(y, y + s);
				cout << "ITER " << ++ITER << endl;
				//cout << rx.end << endl;
				//cout << ry.end << endl;
				Mat curI = I(ry, rx);
				//cout << curI.rows << ',' << curI.cols << endl;
				if (parallel){
					tasks.push_back(thread(myWrapper, curI, E, prm, ry, rx));
				}
				else{
					myWrapper(curI, E, prm, ry, rx);
				}
			}
		}
		if (parallel){
			for (uint i = 0; i < tasks.size(); ++i)
				tasks[i].join();
		}
		toc(start);
	}
	// Fixed this bug! - Aashish Sharma, 20/4/2017. (if maxValue = 0 (no edges found), E contains all NaN values!)
	if (maxValue(E) != 0)
	{
		E = E / maxValue(E);
	}	
}

void myWrapper(const Mat& I, Mat& E, const MyParam& prm, const Range& ry, const Range& rx){
	Detector d(I, prm);
	Mat curE = d.runIm();
	E_mutex.lock();
	E(ry, rx) = max(E(ry, rx), curE);
	E_mutex.unlock();
}


/*
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	mexPrintf("Run Edge Detection\n");
	if (nrhs != 6) {
		mexErrMsgIdAndTxt("MATLAB:mexcpp:nargin", "MEXCPP requires six input arguments.");
	}
	else if (nlhs != 1) {
		mexErrMsgIdAndTxt("MATLAB:mexcpp:nargout", "MEXCPP requires one output argument.");
	}

	if (!mxIsDouble(prhs[0])) {
		mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble", "Input Matrix must be a double.");
	}

	for (int i = 1; i < 6; ++i){
		if (!mxIsDouble(prhs[i]) || mxGetNumberOfElements(prhs[i]) != 1) {
			mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notScalar", "Input multiplier must be a scalar.");
		}
	}

	MyParam prm;
	double* img1 = (double *)mxGetPr(prhs[0]);
	int cols = (int)mxGetN(prhs[0]);
	int rows = (int)mxGetM(prhs[0]);
	mexPrintf(format("Image Size: %d, %d\n", rows,cols).c_str());
	prm.removeEpsilon = mxGetScalar(prhs[1]);
	prm.maxTurn = mxGetScalar(prhs[2]);
	prm.nmsFact = mxGetScalar(prhs[3]);
	prm.splitPoints = (int)mxGetScalar(prhs[4]);
	prm.minContrast = (int)mxGetScalar(prhs[5]);

	mexPrintf(format("Params: %2.2f, %2.2f, %2.2f, %d, %d\n", prm.removeEpsilon, prm.maxTurn, prm.nmsFact, prm.splitPoints, prm.minContrast).c_str());
	Mat I(rows, cols, TYPE);
	memcpy(I.data, img1, I.rows * I.cols * sizeof(double));
	Detector d(I, prm);
	Mat E = d.runIm();
	plhs[0] = mxCreateDoubleMatrix(E.rows, E.cols, mxREAL);
	memcpy(mxGetPr(plhs[0]), E.data, E.rows * E.cols * sizeof(double));
}
*/