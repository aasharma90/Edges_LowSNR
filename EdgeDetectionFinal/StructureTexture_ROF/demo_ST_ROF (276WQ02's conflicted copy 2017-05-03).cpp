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

int main(int argc, char** argv)
{

	Mat I, I_structure, I_texture;
	MyParam prm;

	// Parameters for S-T ROF
	double theta = 0.5; // default = 1/8
	int    nIter = 200;   // default = 100
	double alp   = 0.95;    // default = 0.95

	//Demo for demo noisy image
	cout << "Noisy Image Demo:" << endl;
	I = readImage("../EdgeDetection/Simulations/myCurves2.png");
	I.convertTo(I, TYPE);
	I = I / 255;
	cout << "Image size of " << I.rows << " x " << I.cols << endl;
	// Show the noisy image
	showImage(I, 1, 4, false);
	
	// Perform Structure-Texture decomposition using ROF 
	structure_texture_decomposition_rof(I, I_structure, I_texture, theta, nIter, alp);
	// Show the components
	showImage(I_structure, 2, 4, false);
	showImage(I_texture, 3, 4, true);
}