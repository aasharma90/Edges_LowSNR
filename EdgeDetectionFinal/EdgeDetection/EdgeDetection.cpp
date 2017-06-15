#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "tools.h"
#include "Detector.h"
#include "mex.h"


using namespace std;
using namespace cv;
/*
int main( int argc, char** argv )
{
	if( argc != 2)
    {
		println("Usage: EdgeDetection ImagePath");
		return endRun(-1);
    }

	Mat I;
	if (false){
		cout << argv[1] << endl;
		I = readImage(argv[1]);
		int mid = I.rows / 2+10;
		I(Range(0,mid), Range::all()) = 0;
		I(Range(mid+1, I.rows), Range::all()) = 255;
		resize(I, I, Size(200, 200));
		I.convertTo(I, TYPE);
		I = I / 255;
	}
	else{
		I = imread("real.png", CV_LOAD_IMAGE_GRAYSCALE);
		double a = 0.25;
		//resize(I, I, Size((int)round(I.rows*a),(int)round(I.cols*a)));
		I.convertTo(I, TYPE);
		I = I / 255;
		//showImage(I, 1, 3, true);
		//return 0;
	}

	MyParam prm;
	Detector d(I,prm);
	Mat E = d.runIm();
	E = E / maxValue(E);
	Mat H;
	//hconcat(I, d.getPixelScores(), H);
	//hconcat(H, E, H);
	hconcat(I, E, H);
	//E = E > 0.01;
	showImage(E, 1, 2, true);
	println("Finished");
	return 0;
}

*/
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	mexPrintf("Run Edge Detection\n");
	if (nrhs != 7) {
		mexErrMsgIdAndTxt("MATLAB:mexcpp:nargin", "MEXCPP requires seven input arguments.");
	}
    /*
	else if (nlhs != 1) {
		mexErrMsgIdAndTxt("MATLAB:mexcpp:nargout", "MEXCPP requires one output argument.");
	}

    /*
	if (!mxIsDouble(prhs[0])) {
		mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble", "Input Matrix must be a double.");
	}
    */
    
	for (int i = 1; i < 7; ++i){
		if (!mxIsDouble(prhs[i]) || mxGetNumberOfElements(prhs[i]) != 1) {
			mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notScalar", "Input multiplier must be a scalar.");
		}
	}
    char *string_name;

    int strlen = mxGetN(prhs[0])+1;
    string_name = (char *) mxCalloc(strlen, sizeof(char));
    mxGetString(prhs[0],string_name,strlen);

    mexPrintf("%s\n", string_name);
    
    Mat I = imread(string_name, CV_LOAD_IMAGE_GRAYSCALE);
    I.convertTo(I, TYPE);
    I = I/255;
    int cols = I.cols;
    int rows = I.rows;
	MyParam prm;
	//double* img1 = (double *)mxGetPr(prhs[0]);
	//int cols = (int)mxGetN(prhs[0]);
	//int rows = (int)mxGetM(prhs[0]);
	mexPrintf(format("Image Size: %d, %d\n", rows,cols).c_str());
	prm.removeEpsilon = mxGetScalar(prhs[1]);
	prm.maxTurn = mxGetScalar(prhs[2]);
	prm.nmsFact = mxGetScalar(prhs[3]);
	prm.splitPoints = (int)mxGetScalar(prhs[4]);
	prm.minContrast = (int)mxGetScalar(prhs[5]);
    prm.filterWidth = (int)mxGetScalar(prhs[6]);


	mexPrintf(format("Params: %2.2f, %2.2f, %2.2f, %d, %d, %d\n", prm.removeEpsilon, prm.maxTurn, prm.nmsFact, prm.splitPoints, prm.minContrast, prm.filterWidth).c_str());
	//Mat I(rows, cols, TYPE);
	//memcpy(I.data, img1.data, I.rows * I.cols * sizeof(double));
    
	Detector d(I, prm);
	Mat E = d.runIm();
    E = E / maxValue(E);
    imwrite("res.png",255*E);
    //namedWindow("mainWin", CV_WINDOW_AUTOSIZE);
    //imshow("mainWin", E);
	//plhs[0] = mxCreateDoubleMatrix(E.rows, E.cols, mxREAL);
	//memcpy(mxGetPr(plhs[0]), E.data, E.rows * E.cols * sizeof(double));
}
